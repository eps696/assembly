
import os
import sys
import re
import math
import json
import random
import shutil
import asyncio
import subprocess
import argparse
from pathlib import Path
import librosa
import soundfile

from deep_translator import GoogleTranslator
from langdetect import detect
from langchain_community.document_loaders import TextLoader, PyPDFLoader, UnstructuredWordDocumentLoader, UnstructuredPowerPointLoader

from api_runware import IMAGE_CONFIG, VIDEO_CONFIG, Censored

from util import basename, file_list, txt_clean

trans_en = GoogleTranslator(source='auto', target='en')
TTS_SR = 24000

def base_args(parser=None):
    if parser is None: parser = argparse.ArgumentParser(conflict_handler = 'resolve')
    parser.add_argument('-a',   '--agent',    default='lms', choices=['lms','adk','claude','langchain'], help='Agent backend')
    parser.add_argument('-t',   '--in_txt',   default=None, help='Text string or file for the topic')
    parser.add_argument('-json','--load_json',default=None, help="Load JSON to continue")
    parser.add_argument('-arg', '--load_args',default=None, help="Load saved config")
    parser.add_argument('-evs', '--evals',    default=0, type=int, help="Count of QA evaluation runs for every agent")
    parser.add_argument('-txt', '--txtonly',  action='store_true', help='Text-only mode')
    parser.add_argument('-try', '--max_tries',default=3, type=int, help="Count of generation attempts if failed")
    # paths
    parser.add_argument('-o',    '--out_dir', default="_out", help="Output directory for generated media")
    parser.add_argument('-insd', '--ins_dir', default='data/prompts', help='Instructions directory')
    parser.add_argument('-docs', '--docs',    default=None, help='Source documents')
    parser.add_argument('-tsdd', '--tts_dir', default='models/chatter', help='TTS models directory')
    parser.add_argument('-sndd', '--sound_dir', default='models/mmaudio', help='Sound models directory')
    # vis gen
    parser.add_argument('-tmod', '--txt_model', default=None, help="LLM name")
    parser.add_argument('-imod', '--img_model', default='bfl:5@1', help="Generative model for images")
    parser.add_argument('-vmod', '--vid_model', default='bytedance:2@2', help="Generative model for video")
    parser.add_argument('-iref', '--img_refs',  default=0, type=int, help='Use N reference image[s] for consistency')
    parser.add_argument('-isz', '--img_size', default='1344-752', help="image size, multiple of 8")
    parser.add_argument('-vsz', '--vid_size', default='864-480', help="video size, multiple of 32")
    parser.add_argument('-fps','--fps',     default=24, type=int)
    parser.add_argument('-v',  '--verbose', action='store_true')
    # server
    parser.add_argument('-lh', '--llm_host', default=None, help='LMStudio server host')
    parser.add_argument('--runware_api_key', default=None)
    parser.add_argument('--poll_interval', default=2.0, type=float)
    parser.add_argument('--poll_timeout', default=300, type=int)
    # Backend-specific args (each backend uses what it needs)
    parser.add_argument('--db_url', default=None, help='[ADK] Database URL')
    parser.add_argument('--cld_cache', default=True, action='store_true', help='[Claude] Prompt caching')
    parser.add_argument('--use_thinking', default=False, action='store_true', help='[Claude] Extended thinking')
    parser.add_argument('--thinking_budget', default=5000, type=int, help='[Claude] Thinking budget')
    parser.add_argument('--use_graph',  action='store_true', help='[LangChain] Use LangChain message loop or LangGraph StateGraph with checkpointer')
    return parser

def get_agent(state, a, defs):
    """Factory function to create the appropriate agent backend"""
    if a.agent == 'lms':
        from agt_llm import AgentLLM
        return AgentLLM(state, a, a.ins_dir)
    elif a.agent == 'adk':
        from agt_adk import AgentADK
        return AgentADK(state, a, a.ins_dir, defs, db_url=getattr(a, 'db_url', None))
    if a.agent == 'claude':
        from agt_claude import AgentClaude
        return AgentClaude(state, a, a.ins_dir, defs)
    elif a.agent == 'langchain':
        from agt_lang import AgentLang
        return AgentLang(state, a, a.ins_dir, defs)
    else:
        raise ValueError(f"Unknown agent type: {a.agent}")

def astr2caps(text):
    """Convert asterisk-wrapped text to uppercase"""
    return re.sub(r'\*([^*]+)\*', lambda m: m.group(1).upper(), text)

def time2frames(seconds, fps=25, type='comfy'):
    """Convert seconds to frame count for different backends"""
    if 'comfy' in type:
        return 8 * round(seconds * fps / 8) + 1 # nearest multiple of 8, then add 1
    elif 'wan' in type:
        return math.ceil((seconds * fps - 9) / 12) * 12 + 9 # 12k*9
    else:  # latwalk
        return round(seconds * fps)

def upda(a, **kwargs):
    """Update a object in place"""
    a.__dict__.update(kwargs) # updates in place
    # for k, v in kwargs.items(): setattr(a, k, v)
    return a

def parse_json(text):
    """Extract and parse JSON from LLM output"""
    json_str = text[text.find('{'):text.rfind('}')+1] # get json
    json_str = ''.join(c for c in json_str if ord(c) >= 32 or c in '\n\r').replace('\u2011', '-') # clean chars
    json_str = ' '.join(json_str.split()) # clean spaces
    return json.loads(json_str)

class DurationError(Exception):
    """Raised when audio duration is out of acceptable range"""
    pass

class PromptMan:
    """Manages prompt templates from markdown files"""
    def __init__(self, prompts_dir):
        self.prompts_dir = prompts_dir
        self.prompts = {}

    def get_prompt(self, name):
        """Load prompt from file (cached). Falls back to parent prompts directory for universal prompts."""
        if name not in self.prompts:
            filepath = os.path.join(self.prompts_dir, f"{name}.md")
            # Fallback to parent prompts directory for universal prompts (eval, etc.)
            if not os.path.isfile(filepath):
                parent_dir = os.path.dirname(self.prompts_dir)
                filepath = os.path.join(parent_dir, f"{name}.md")
            if os.path.isfile(filepath):
                with open(filepath, 'r', encoding='utf-8') as f:
                    self.prompts[name] = f.read()
            else:
                print(f" !! Prompt not found: {name}.md")
                self.prompts[name] = ""
        return self.prompts[name]

class StoryState:
    """Centralized state management for story generation (async-safe)"""

    def __init__(self, data, logfile):
        self.data = {k: v.copy() if isinstance(v, (list, dict)) else v for k, v in data.items()}
        self.logfile = logfile
        self.current_chapter = 1
        self.lock = asyncio.Lock()

    async def merge_data(self, new_data):
        """Merge new data into existing state (async-safe)"""
        async with self.lock:
            for key, value in new_data.items():
                if key not in self.data:
                    continue

                if isinstance(value, list): # list merging
                    if self.data[key] and value and isinstance(value[0], dict):
                        id_fields = [k for k in value[0].keys() if 'number' in k or 'name' in k]
                        if id_fields: # named/numbered items
                            for item in value:
                                found = False
                                for i, existing in enumerate(self.data[key]):
                                    if isinstance(existing, dict) and all(existing.get(f) == item.get(f) for f in id_fields):
                                        self.data[key][i] = item  # update existing
                                        found = True
                                        break
                                if not found:
                                    self.data[key].append(item) # add new
                        else:
                            self.data[key].extend(value) # generic items
                    else:
                        self.data[key].extend(value) # non-dict lists or empty lists

                elif isinstance(value, dict): # nested dictionary merging
                    if not isinstance(self.data[key], dict):
                        self.data[key] = value
                    else:
                        for nest_key, nest_value in value.items():
                            if nest_key in self.data[key] and isinstance(self.data[key][nest_key], list):
                                if isinstance(nest_value, list): # Both are lists - merge them
                                    if nest_value and isinstance(nest_value[0], dict):
                                        self.data[key][nest_key].extend(nest_value) # list of dicts - just extend
                                    else:
                                        for item in nest_value: # simple list - avoid duplicates
                                            if item not in self.data[key][nest_key]:
                                                self.data[key][nest_key].append(item)
                                else:
                                    if nest_value not in self.data[key][nest_key]: # nest_value is an element - append it
                                        self.data[key][nest_key].append(nest_value)
                            else:
                                self.data[key][nest_key] = nest_value # replace scalars or set new keys
                else:
                    self.data[key] = value

    async def save(self, filepath=None):
        """Save state to JSON file"""
        if not filepath: filepath = self.logfile
        async with self.lock:
            Path(filepath).parent.mkdir(parents=True, exist_ok=True)
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(self.data, f, indent=2, ensure_ascii=False)

    async def load(self, filepath):
        """Load state from JSON file"""
        async with self.lock:
            with open(filepath, 'r', encoding='utf-8') as f:
                data = json.load(f)
            self.data = data

def load_docs(src, copy=None):
    docs = []
    filelist = file_list(src) if os.path.isdir(src) else [src] if os.path.isfile(src) else []
    if copy: 
        os.makedirs(copy, exist_ok=True)
        for f in filelist: shutil.copy2(f, os.path.join(copy, os.path.basename(f)))
    loaders = {"txt": TextLoader, "pdf": PyPDFLoader, "docx": UnstructuredWordDocumentLoader, "pptx": UnstructuredPowerPointLoader}
    for path in filelist:
        if not os.path.isfile(path): continue
        print(path)
        ext = os.path.splitext(path.lower())[-1][1:]
        load_kwarg = {'encoding': "utf-8"} if ext=='txt' else {}
        if ext in loaders:
            try:
                pages = loaders[ext](path, **load_kwarg).load()
                if not isinstance(pages, list): pages = [pages]
                content = '\n'.join([page.page_content for page in pages])
                if detect(content) != 'en':
                    ll = 3000
                    content = ''.join([trans_en.translate(content[n*ll:(n+1)*ll]) for n in range(len(content) // ll)])
                docs.append({"filename": basename(path), "content": content})
            except Exception as e:
                print(f"[!] {basename(path)}: {e}")
    print(f".. loaded {len(docs)} docs", [str(len(x["content"])//1024) + 'k' for x in docs])
    return docs

def mix_img_wav(img, wav, out_path, size, bitrate=None, fps=None, tts_sr=TTS_SR, timeout=60):
    """Create video from image + audio using ffmpeg"""
    command = ['ffmpeg', '-y', '-v', 'error', '-hwaccel', 'cuda']
    command.extend(['-loop', '1', '-framerate', str(fps or 25)]) # loop image and specify framerate before input
    command.extend(['-i', img, '-i', wav, '-c:v', 'h264_nvenc', '-preset', 'medium'])
    if bitrate is not None:
        command.extend(['-b:v', f'{bitrate}M'])
    command.extend(['-profile:v', 'high', '-c:a', 'aac', '-b:a', '192k', 
                    '-ar', str(tts_sr), '-ac', '1', '-s', size, '-pix_fmt', 'yuv420p', '-shortest', out_path])
    try:
        result = subprocess.run(command, capture_output=True, timeout=timeout, text=True)
        if result.returncode != 0:
            raise RuntimeError(f"FFmpeg failed: {result.stderr}")
    except subprocess.TimeoutExpired:
        raise RuntimeError(f"FFmpeg timeout after {timeout}s")

def mix_mov_wav(mov, wav, out_path, tts_sr=TTS_SR, timeout=120):
    """Add audio to video using ffmpeg"""
    command = ['ffmpeg', '-y', '-v', 'error', '-i', mov, '-i', wav,
               '-c:v', 'copy', '-c:a', 'aac', '-b:a', '192k', '-ar', str(tts_sr), '-ac', '1', out_path]
    try:
        result = subprocess.run(command, capture_output=True, timeout=timeout, text=True)
        if result.returncode != 0:
            raise RuntimeError(f"FFmpeg failed: {result.stderr}")
    except subprocess.TimeoutExpired:
        raise RuntimeError(f"FFmpeg timeout after {timeout}s")

class MediaGen:
    """Async media generation for TTS and visuals"""

    def __init__(self, a):
        self.a = a
        self.tts = None
        self.rw = None
        self.mma = None
        self.refs = {}
        self.init_vis()

    async def init_vis(self):
        """Async initialization of visual/audio backends"""
        a = self.a
        try:
            import torch
            from sound import MMAud
            from tts import ChatterBox
            from api_runware import RunwareGen
            self.tts = ChatterBox(a.tts_dir)
            self.mma = MMAud(model_dir=a.sound_dir, model_name='large_44k_v2')
            self.rw = RunwareGen(a)
            print("Media generation initialized")
        except Exception as e:
            print(f"Warning: Media generation init failed: {e}")

    def descript_(self, subject):
        """Parse subject into name and look"""
        if isinstance(subject, str):
            if ':' in subject:
                [name, look] = subject.split(':')
            else: # ',' in subject:
                splits = subject.split(',')
                [name, look] = splits[0], ','.join(splits[1:])
        else: # isinstance(subject, dict):
            name, look = subject.get('name', ''), subject.get('look', '')
        return name, look

    async def gen_ref_(self, subject, type, ref_dir=None):
        """Generate reference image for subject"""
        name, look = self.descript_(subject)
        if not name: return

        if ref_dir is None: 
            ref_dir = os.path.join(self.a.out_dir, 'refs')
        fname = os.path.join(ref_dir, '%s-%s' % (type, txt_clean(name)))
        if os.path.exists(fname + '-rw.png'):
            self.refs[name] = fname + '-rw.png'
            print(f"Ref for {type} '{name}' exists")
            return

        if 'act' in type:
            ref_prompt = f"Professional photograph of {name}: {look}. Neutral background, consistent lighting."
        else: # 'loc' in type:
            ref_prompt = f"Establishing shot of {name}: {look}. Clear architectural features, distinctive environment, consistent lighting."

        try:
            ref_img = await asyncio.to_thread(self.rw.gen_t2i, fname, ref_prompt, basename(ref_dir))
            self.refs[name] = ref_img
            print(f" Ref for {type} {name} new")
        except Censored as e:
            print(f" !! Ref for {type} '{name}' SKIPPED - moderation rejection")
        except Exception as e:
            print(f" !! Ref for {type} '{name}' FAILED: {e}")

    async def gen_refs(self, actors, locations):
        """Generate reference images for all actors and locations"""
        for actor in actors:
            await self.gen_ref_(actor, 'actor')
        # for location in locations:
            # await self.gen_ref_(location, 'locat')

    async def get_scene_refs(self, scene_data):
        """Get relevant reference images for a scene"""
        scene_refs = []
        glob_actors = scene_data.get('global_actors', [])
        for actor in glob_actors:
            actor_name, _ = self.descript_(actor)
            if actor_name and actor_name in self.refs:
                scene_refs.append(self.refs[actor_name])
        # glob_setting = scene_data.get('global_setting', '')
        # for loc_name in self.refs.keys():
            # if loc_name in glob_setting:
                # scene_refs.append(self.refs[loc_name])
        local_actors = scene_data.get('local_actors', [])
        for actor in local_actors:
            actor_name, _ = self.descript_(actor)
            if actor_name:
                if actor_name not in self.refs:
                    await self.gen_ref_(actor, 'actor') # gen ref for local actor
                if actor_name in self.refs:
                    scene_refs.append(self.refs[actor_name])
        # local_settings = scene_data.get('local_settings', [])
        # for setting in local_settings:
            # set_name, _ = self.descript_(setting)
            # if set_name:
                # if set_name not in self.refs:
                    # self.gen_ref_(setting, 'locat') # gen ref for local setting
                # if set_name in self.refs:
                    # scene_refs.append(self.refs[set_name])
        return scene_refs if scene_refs else None

    async def gen_voice(self, fname, text, speaker):
        """Generate voice from text"""
        wav_path = fname + '.wav'
        text = astr2caps(text)
        try:
            try:
                await asyncio.to_thread(self.tts.tts_to_file, text=text, speaker=speaker, file_path=wav_path)
            except Exception as e:
                print(f"!! TTS failed: {e}")
                raise

            duration = librosa.get_duration(path=wav_path)
            cfg = self.rw._get_vid_config()
            min_dur, max_dur = cfg['duration']
            if not (min_dur <= duration <= max_dur):
                os.remove(wav_path)  # Clean up invalid audio
                raise DurationError(f"Audio {duration:.1f}s not in {cfg['duration']}")

            return {"status": "success", "wav_path": wav_path, "speaker": speaker, "text": text}

        except DurationError:
            raise  # Propagate to caller for retry
        except Exception as e:
            return {"status": "error", "message": str(e)}

    async def gen_visual(self, fname, txt_vis, scene_data=None):
        """Generate visual media (image + video)"""
        wav_path = fname + '.wav'
        out_path = fname + '.mp4'

        ref_dict = {}
        if self.a.img_refs > 0 and scene_data and self.refs:
            scene_refs = await self.get_scene_refs(scene_data)
            if scene_refs:
                ref_dict['ref_images'] = scene_refs[:self.a.img_refs]
        try:
            try:
                img_path = await asyncio.to_thread(self.rw.gen_t2i, fname, txt_vis, **ref_dict)
            except Censored as e:
                return {"status": "censored", "message": str(e), "stage": "t2i"}
            except Exception as e:
                print(f"!! T2I failed, skipping: {e}")
                return {"status": "error", "message": str(e), "stage": "t2i"}

            duration = librosa.get_duration(path=wav_path)
            try:
                mov_path = await asyncio.to_thread(self.rw.gen_i2v, fname, txt_vis, duration, [img_path])
                output = {"status": "success", "video_path": out_path}
            except Censored as e:
                mov_path = None
                output = {"status": "partial", "message": str(e), "video_path": out_path}
            except Exception as e:
                mov_path = None
                output = {"status": "partial", "message": str(e), "video_path": out_path}
                print(f"!! I2V failed, using static image !!") # {e}

            if mov_path and mov_path.endswith('.mp4'):
                await asyncio.to_thread(mix_mov_wav, mov_path, wav_path, out_path)
            else:
                img_for_video = img_path['path'] if isinstance(img_path, dict) else img_path
                await asyncio.to_thread(mix_img_wav, img_for_video, wav_path, out_path, size=self.a.vid_size, fps=self.a.fps)
            return output

            return {"status": "success", "video_path": out_path}
        except Exception as e:
            return {"status": "error", "message": str(e)}

