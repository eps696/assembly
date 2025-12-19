
import os
import json
import asyncio
import shutil
import argparse
from pathlib import Path

from base import MediaGen, StoryState, DurationError, load_docs, base_args, sdfu_args
from util import Tm, filter_items, max_num, fnexist, rand_pick, file_list, save_cfg

def get_args(parser=None):
    """LMStudio-specific arguments"""
    if parser is None: parser = argparse.ArgumentParser(conflict_handler = 'resolve')
    # overwrite
    parser.add_argument('-insd', '--ins_dir', default='data/prompts/story', help='Instructions directory')
    return parser.parse_args()

DEFAULT_DATA = {
    "global_settings": {},
    "global_actors": [],
    "chapters": [],
    "scenes": [],
    "fragments": [],
    "visuals": [],
    "voices": []
}

agent_defs = {
    'arch_init': {'desc': "Initialize global story structure, settings, and characters", 'tools': ['search']},
    'writ_chap': {'desc': "Write chapter content", 'tools': ['search']},
    'writ_scen': {'desc': "Create scenes within chapters", 'tools': ['search']},
    'writ_frag': {'desc': "Write narrative fragments for scenes"},
    'writ_voc': {'desc': "Create voice-over narration"},
    'writ_vis': {'desc': "Generate visual scene descriptions"},
    'arch_upd': {'desc': "Update global story structure based on new content", 'tools': ['search']}
}

async def gen_chapter(chapter_num, idx, a, ai, state, mediagen=None):
    """Generate all content for a single chapter"""
    print(f"\n{'='*60}")
    print(f"CHAPTER {chapter_num}")
    tm = Tm()

    speakrs = ['f-227','f-245','m-229','m-230','f-257','f-270','m-236','m-262','f-294','f-306','m-286','m-317','f-311','f-364']
    speakr = speakrs[0]
    cur_vis_prev = []
    do_media = not a.txtonly and mediagen is not None

    # chapter_num
    cur_chapters = filter_items(state.data["chapters"], chapter_number=chapter_num)
    if not cur_chapters:
        cur_input = {
            "global_settings": state.data["global_settings"],
            "global_actors": state.data["global_actors"],
            "chapters": state.data["chapters"]
        }
        cur_chapters = await ai.call_agent(cur_input, 'writ-chap', 'chapters')
    cur_chapter = cur_chapters[0]
    if a.verbose: print(tm.do('chap %d' % cur_chapter['chapter_number']))

    # scenes for chapter_num
    cur_scenes = filter_items(state.data["scenes"], chapter_number=chapter_num)
    if not cur_scenes:
        cur_input = {
            "global_settings": state.data["global_settings"],
            "global_actors": state.data["global_actors"],
            "chapter": cur_chapter
        }
        cur_scenes = await ai.call_agent(cur_input, 'writ-scen', 'scenes')
    scene_count = max_num(cur_scenes, "scene_number")
    if a.verbose: print(tm.do('scenes %d' % scene_count))

    for scene_num in range(1, scene_count + 1):
        # fragments for scene_num
        cur_scene = filter_items(cur_scenes, scene_number=scene_num)[0]
        if a.verbose: print('scen', cur_chapter['chapter_number'], cur_scene['scene_number'])

        cur_frags = filter_items(state.data["fragments"], chapter_number=chapter_num, scene_number=scene_num)
        if not cur_frags:
            cur_input = {"scene": cur_scene}
            cur_frags = await ai.call_agent(cur_input, 'writ-frag', 'fragments')
        frag_count = max_num(cur_frags, "fragment_number")
        if a.verbose: print(tm.do('frags %d' % frag_count))

        for frag_num in range(1, frag_count + 1):
            cur_fragment = filter_items(cur_frags, fragment_number=frag_num)[0]
            if a.verbose: print('frag', cur_chapter['chapter_number'], cur_scene['scene_number'], cur_fragment['fragment_number'])

            # multimedia for frag_num
            speakr = rand_pick(speakrs, speakr)
            fmask = os.path.join(a.out_dir, f"{idx:03d}-{chapter_num:02d}_{scene_num:02d}_{frag_num}")
            fname = fmask + f"-{speakr}"
            if not fnexist(fmask, 'mp4'):

                # voice
                voc_result = None
                for voc_attempt in range(a.max_tries):
                    cur_voc = filter_items(state.data["voices"], chapter_number=chapter_num, scene_number=scene_num, fragment_number=frag_num)
                    if voc_attempt > 0 or not cur_voc:
                        cur_input = {"fragment": cur_fragment}
                        cur_voc = await ai.call_agent(cur_input, 'writ-voc', 'voices')
                        if a.verbose: print(tm.do('voice'))
                    cur_voc = cur_voc[0]
                    # tts
                    if do_media:
                        try:
                            voc_result = await mediagen.gen_voice(fname, cur_voc['content'], speakr)
                            if a.verbose: print(tm.do('tts'))
                            break # success
                        except DurationError as e:
                            if voc_attempt < a.max_tries - 1:
                                if a.verbose: print(f"!! invalid duration, remaking voice-over (attempt {voc_attempt+1}/{a.max_tries})...")
                            else:
                                print(f"!! wrong duration after {a.max_tries} attempts, skipping fragment")
                                voc_result = {"status": "error", "message": str(e)}
                                break
                    else: break

                # visual
                cur_vis = filter_items(state.data["visuals"], chapter_number=chapter_num, scene_number=scene_num, fragment_number=frag_num)
                if not cur_vis:
                    cur_input = {
                        "global_settings": state.data["global_settings"],
                        "global_actors": state.data["global_actors"],
                        "chapter": cur_chapter,
                        "scene": cur_scene,
                        "fragment": cur_fragment
                    }
                    cur_vis = await ai.call_agent(cur_input, 'writ-vis', 'visuals')
                    if a.verbose: print(tm.do('previs'))
                cur_vis = cur_vis[0]

                if do_media and voc_result and voc_result['status'] == 'success':
                    prompt_fix = False
                    cur_vis_orig = cur_vis
                    for vis_attempt in range(a.max_tries):
                        prompts = cur_vis['content']
                        vis_result = await mediagen.gen_visual(fname, prompts, scene_data=cur_scene)
                        if vis_result.get('status') != 'censored':
                            if a.verbose: print(tm.do('vis'))
                            break # success or partial - exit retry loop

                        if vis_attempt < a.max_tries - 1:
                            print(f" .. Fragment {idx} rejected, fix {vis_attempt + 1} ..")
                            fix_input = {
                                "rejected_prompt": cur_vis_orig['content'],
                                "error_message": vis_result.get('message', ''),
                                "scene_context": cur_scene.get('content', ''),
                                "fragment_context": cur_fragment.get('content', '')
                            }
                            vis_fix = await ai.call_agent(fix_input, 'fix-vis', 'visuals', save=False)
                            cur_vis['content'] = vis_fix[0]['content']
                            prompt_fix = True
                        else:
                            print(f" !! Fragment {idx} SKIPPED after {a.max_tries} moderation rejections")

                    cur_vis_prev = [cur_vis['content']]
                    if vis_result.get('status') == 'success':
                        if prompt_fix:
                            await state.merge_data({"visuals": [cur_vis]})
                            await state.save()
                        if mediagen.mma and mediagen.mma.ok:
                            snd_prompt = "Sound effects for the scene: %s. NO SPEECH!" % cur_fragment['content']
                            await asyncio.to_thread(mediagen.mma.gen, vis_result['video_path'], snd_prompt, volume=0.27)
                            if a.verbose: print(tm.do('snd'))
            idx += 1

    # Update globals
    next_chapter = filter_items(state.data["chapters"], chapter_number=chapter_num + 1)
    if not next_chapter:
        cur_input = {
            "global_settings": state.data["global_settings"],
            "global_actors": state.data["global_actors"],
            "chapters": filter_items(state.data["chapters"], chapter_number=list(range(1, chapter_num + 1))),
            "scenes": cur_scenes
        }
        result = await ai.call_agent(cur_input, 'arch-upd', save=False)
        if "global_actors" in result:
            existing_names = {act.get("name") for act in state.data["global_actors"]}
            for actor in result["global_actors"]:
                if actor.get("name") not in existing_names:
                    state.data["global_actors"].append(actor)
        if a.img_refs > 0 and not a.txtonly and mediagen:
            await mediagen.gen_refs(state.data['global_actors'], state.data['global_settings'].get('locations', []))
        await state.save()
        if a.verbose: print(tm.do('update'))


async def main():
    """Main execution function"""
    a = get_args(base_args())
    tm = Tm()

    load_json = a.load_json # save from overriding
    if a.load_args and os.path.isfile(a.load_args):
        a = load_args(a.load_args, a)

    save_cfg(a, a.out_dir)
    logfile = os.path.join(a.out_dir, 'log.json')
    db_url = 'sqlite:///' + os.path.join(a.out_dir, 'sessions.db')

    state = StoryState(DEFAULT_DATA, logfile)
    if load_json and os.path.isfile(load_json):
        await state.load(load_json)

    # Initialize media generator
    mediagen = None
    if not a.txtonly:
        print('..loading media gen')
        a.size = a.vid_size
        mediagen = MediaGen(a)
        await mediagen.init_vis()

    # Select agent backend
    if 'adk' in a.agent:
        from agt_adk import AgentADK
        ai = AgentADK(state, a, a.ins_dir, agent_defs, db_url)
    elif 'lms' in a.agent:
        from agt_llm import AgentLLM
        ai = AgentLLM(state, a, a.ins_dir)
    print(tm.do('setup'))

    # Init story
    if state.data['global_settings'] and state.data['global_actors']:
        await state.save()
    else:
        init_input = {}
        if a.doc_dir:
            init_input["documents"] = load_docs(a.doc_dir, copy_dir=os.path.join(srcdir, 'docs'))
        if a.in_txt:
            init_input["initial_concept"] = open(a.in_txt, 'r', encoding="utf-8").read() if os.path.isfile(a.in_txt) else a.in_txt
        await ai.call_agent(init_input, 'arch-init', 'global_actors')
        print(tm.do('arch_init'))

    # Generate reference images
    if a.img_refs > 0 and not a.txtonly and mediagen:
        await mediagen.gen_refs(state.data['global_actors'], state.data['global_settings'].get('locations', []))
        print(tm.do('refs'))

    try:
        idx = 0
        for chapnum in range(1, 9999):
            idx = await gen_chapter(chapnum, idx, a, ai, state, mediagen)
            print(tm.do(f'Chapter {chapnum} completed'))

    except KeyboardInterrupt:
        print("\n\nInterrupted by user")
    except Exception as e:
        print(f"\n\nError: {e}")

    await state.save()
    print(f"State saved to {logfile}")


if __name__ == '__main__':
    asyncio.run(main())
