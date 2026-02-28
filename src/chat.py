
import os
import sys
import json
import asyncio
import argparse
from itertools import cycle

from base import MediaGen, StoryState, DurationError, get_agent, load_docs, base_args
from util import Tm, filter_items, fnexist, fuzzy_find, rand_pick, load_args, file_list, save_cfg

def get_args(parser=None):
    if parser is None: parser = argparse.ArgumentParser(conflict_handler = 'resolve')
    parser.add_argument('-pers', '--pers_mode', default='fix', help='Personas treatment - fix (fixed), evo (evolving), anti (provocative)')
    parser.add_argument(       '--maxmsg',  default=6, type=int, help='Amount of recent fragments for context')
    parser.add_argument(       '--updmsg',  default=8, type=int, help='Update personas every N fragments')
    parser.add_argument('-n',  '--num',     default=999, type=int, help='Number of fragments to generate')
    parser.add_argument('-np', '--num_personas', default=3, type=int, help='Number of debate personas')
    # overwrite default
    parser.add_argument('-insd', '--ins_dir', default='data/prompts/chat', help='Instructions directory')
    return parser.parse_args()

start_prompt = 'The world is obscure and incomprehensible. Discuss this topic.'

def get_data_schema(mode):
    DEFAULT_DATA = {
        "global_settings": {},
        "personas": [],
        "fragments": [],
        "voices": [],
        "visuals": []
    }
    agent_defs = {
        'arch_init': {'desc': "Initialize debate personas and settings", 'tools': ['search']},
        'writ_frag': {'desc': "Write elaborated inner thoughts for the current persona's turn"},
        'writ_voc': {'desc': "Condense thoughts into short spoken remark (max 30 words)"},
        'writ_vis': {'desc': "Generate visual description for the fragment"}
    }

    if mode and mode.lower()=='anti': # provocative
        DEFAULT_DATA["directions"] = [] # track thought directions for analysis
        DEFAULT_DATA["provocation_modes"] = [] # recent modes for rotation tracking
        agent_defs = {**agent_defs,
            'curat_dir_anti': {'desc': "Analyze debate and select provocative direction", 'tools': ['search']},
            'curat_pers_anti': {'desc': "Select or create persona for controversial position"}
        }
    elif mode and mode.lower()=='evo': # evolving
        DEFAULT_DATA["directions"] = [] # track thought directions for analysis
        agent_defs = {**agent_defs,
            'curat_dir_evo': {'desc': "Analyze debate and forecast next thought direction", 'tools': ['search']},
            'curat_pers_evo': {'desc': "Select or create persona for thought direction"},
        }
    else: # fix / default
        agent_defs['arch_upd'] = {'desc': "Update persona states based on debate progress", 'tools': ['search']}

    return DEFAULT_DATA, agent_defs

def get_active_personas(personas):
    return [p for p in personas if p.get('active', True)]

def assign_speaker(persona, pools):
    """Assign TTS speaker based on gender"""
    gender = persona.get("gender", "m")
    if gender == "n":  # non-human: alternate between pools
        pool_key = 'm' if len([p for p in persona.get('name', '')]) % 2 == 0 else 'f'
    else:
        pool_key = gender[0].lower()
    return next(pools.get(pool_key, pools['m']))

async def gen_debate(idx, a, ai, state, mediagen=None, cur_vis_prev=[], speaker_pools=None):
    """Generate single debate fragment with voice and visual"""
    tm = Tm()
    do_media = not a.txtonly and mediagen is not None

    ctx_cache = {'context': {"global_settings": state.data["global_settings"]}} # update if settings updated??

    # get or gen fragment
    cur_frag = filter_items(state.data["fragments"], fragment_number=idx)
    if not cur_frag:
        recent_frags = state.data["fragments"][-a.maxmsg:] if state.data["fragments"] else [] # context = last K thoughts
        # recent_frags = state.data["voices"][-a.maxmsg:] if state.data["voices"] else [] # context = last K sayings

        if a.pers_mode=='fix':
            last_pers_name = state.data["fragments"][-1].get("persona_name") if state.data["fragments"] else None
            last_pers = fuzzy_find(state.data["personas"], last_pers_name)
            cur_pers = rand_pick(state.data["personas"], last_pers) # random, avoiding repeat
            cur_input = {
                "fragment_number": idx,
                "persona": cur_pers,
                "recent_fragments": recent_frags
            }
            cur_frag = await ai.call_agent(cur_input, 'writ-frag', 'fragments', **ctx_cache)

        else:
            personas = get_active_personas(state.data["personas"])

            # thought direction
            cur_input = {
                "personas": personas,
                "recent_fragments": recent_frags
            }
            if 'anti' in a.pers_mode:
                cur_input["recent_provocation_modes"] = state.data.get("provocation_modes", [])[-5:] # last 5 
                cur_instr = 'curat-dir-anti'
            else:
                cur_instr = 'curat-dir-evo'
            direction = await ai.call_agent(cur_input, cur_instr, **ctx_cache, save=False)
            cur_dir = direction.get("direction", {})
            cur_dir["fragment_number"] = idx
            state.data["directions"].append(cur_dir)
            thought_seed = cur_dir.get("thought_seed", "")
            if a.verbose: print(tm.do('direction'))

            # make persona
            last_speaker_name = recent_frags[-1].get("persona_name") if recent_frags else None
            cur_input = {
                "direction": cur_dir,
                "personas": personas,
                "recent_fragments": recent_frags[-3:], # shorter context for persona selection
                "last_speaker_name": last_speaker_name
            }
            cur_instr = 'curat-pers-anti' if 'anti' in a.pers_mode else 'curat-pers-evo'
            result = await ai.call_agent(cur_input, cur_instr, **ctx_cache, save=False)
            selection = result.get("selection", {})
            cur_pers = result.get("persona", {})
            
            # new persona?
            add_pers = selection.get("action") == "create_new"
            if not add_pers:
                pers_name = selection.get("persona_name") or cur_pers.get("name")
                existing = fuzzy_find(state.data["personas"], pers_name)
                if existing:
                    cur_pers = existing
                elif cur_pers.get("name"):
                    add_pers = True # treat as new
                else:
                    cur_pers = state.data["personas"][0]
                    if a.verbose: print(f" PERS fallback: {cur_pers.get('name', '?')}")
            if add_pers:
                if do_media and a.img_refs > 0:
                    await mediagen.gen_refs([cur_pers], []) # state.data['global_settings'].get('locations', [])
                if speaker_pools:
                    cur_pers["speaker"] = assign_speaker(cur_pers, speaker_pools)
                state.data["personas"].append(cur_pers)
                if a.verbose: print(f" PERS NEW: {cur_pers.get('name')} ({cur_pers.get('nature', '???')})")
            if a.verbose: print(tm.do('persona'))

            # fragment with thought seed
            cur_input = {
                "fragment_number": idx,
                "persona": cur_pers,
                "thought_seed": thought_seed,
                "recent_fragments": recent_frags
            }
            cur_frag = await ai.call_agent(cur_input, 'writ-frag', 'fragments', **ctx_cache)
            if cur_pers.get("name"):
                cur_frag[0]["persona_name"] = cur_pers["name"]

    else:
        pers_name = cur_frag[0].get("persona_name") # from existing fragment
        cur_pers = fuzzy_find(state.data["personas"], pers_name) or state.data["personas"][0]

    cur_frag = cur_frag[0]
    if a.verbose: print(tm.do('frag %d' % idx))

    speakr = cur_pers.get("speaker", "p230")

    # Media generation
    fmask = os.path.join(a.out_dir, f"{idx:03d}")
    fname = fmask + f"-{speakr}"
    if not fnexist(fmask, 'mp4'):

        # voice-over
        async def gen_voc():
            """Generate voice-over text with TTS retry loop"""
            for voc_attempt in range(a.max_tries):
                cur_voc = filter_items(state.data["voices"], fragment_number=idx)
                if voc_attempt > 0 or not cur_voc:
                    cur_input = {"fragment": cur_frag, "persona": cur_pers, "fragment_number": idx}
                    cur_voc = await ai.call_agent(cur_input, 'writ-voc', 'voices')
                cur_voc = cur_voc[0]
                if cur_pers.get("name"):
                    cur_voc["persona_name"] = cur_pers["name"]
                # tts
                if do_media:
                    try:
                        voc_result = await mediagen.gen_voice(fname, cur_voc['content'], speakr) # generate voiceover
                        return voc_result
                    except DurationError as e:
                        if voc_attempt < a.max_tries - 1:
                            if a.verbose: print(f"!! invalid duration, retry voice-over ({voc_attempt+1}/{a.max_tries})...") # {e}
                        else:
                            print(f"!! wrong duration after {a.max_tries} attempts, skipping fragment")
                            return {"status": "error", "message": str(e)}
                else: return None
            return None
        # visual
        async def gen_vis_text():
            cur_vis = filter_items(state.data["visuals"], fragment_number=idx)
            if not cur_vis:
                cur_input = {"fragment": cur_frag, "persona": cur_pers, "fragment_number": idx}
                cur_vis = await ai.call_agent(cur_input, 'writ-vis', 'visuals', **ctx_cache)
            cur_vis = cur_vis[0]
            if cur_pers.get("name"):
                cur_vis["persona_name"] = cur_pers["name"]
            return cur_vis

        voc_result, cur_vis = await asyncio.gather(gen_voc(), gen_vis_text())
        if a.verbose: print(tm.do('voc+vis'))

        # video
        if do_media and voc_result and voc_result.get('status') == 'success':
            prompts = cur_vis['content']
            scene_data = {"global_actors": [cur_pers]}  # for ref image lookup
            vis_result = await mediagen.gen_visual(fname, prompts, scene_data=scene_data)
            if a.verbose: print(tm.do('vis'))
            cur_vis_prev = [cur_vis['content']]

    # Periodic persona update
    if a.pers_mode == 'fix' and idx > 0 and idx % a.updmsg == 0:
        upd_input = {
            "personas": state.data["personas"],
            "recent_fragments": state.data["fragments"][-a.updmsg:]
        }
        result = await ai.call_agent(upd_input, 'arch-upd', **ctx_cache, save=False)
        # Merge updated personas by name
        if "personas" in result:
            for upd_persona in result["personas"]:
                match = fuzzy_find(state.data["personas"], upd_persona.get("name"))
                if match:
                    i = state.data["personas"].index(match)
                    upd_persona["speaker"] = match.get("speaker")
                    state.data["personas"][i] = upd_persona
        if "global_settings" in result:
            state.data["global_settings"].update(result["global_settings"])
        await state.save()
        if a.verbose: print(tm.do('update'))

    return idx + 1, cur_vis_prev


async def main():
    """Main execution function"""
    a = get_args(base_args())
    tm = Tm()

    load_json = a.load_json # save from overriding
    if a.load_args and os.path.isfile(a.load_args):
        a = load_args(a.load_args, a)

    save_cfg(a, a.out_dir)
    logfile = os.path.join(a.out_dir, 'log.json')

    DEFAULT_DATA, agent_defs = get_data_schema(a.pers_mode)

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
    ai = get_agent(state, a, agent_defs)
    print(tm.do(f'.. {a.agent} setup'))

    # Get initial topic
    if a.in_txt:
        init_topic = open(a.in_txt, 'r', encoding="utf-8").read() if os.path.isfile(a.in_txt) else a.in_txt
    else:
        init_topic = start_prompt
    if a.docs:
        init_input["documents"] = load_docs(a.doc_dir, copy_dir=os.path.join(srcdir, 'docs'))

    sp_f = ['f-245','f-311']
    sp_m = ['m-262','m-286']
    speaker_pools = {'m': cycle(sp_m), 'f': cycle(sp_f)}

    # Init personas (once)
    if not state.data['personas']:
        num_pers = a.num_personas if a.pers_mode == 'fix' else 1
        init_input = {"initial_topic": init_topic, "num_personas": num_pers}
        cur_instr = 'arch-init' if 'fix' in a.pers_mode else 'arch-init-evo'
        await ai.call_agent(init_input, cur_instr, 'personas')
        print(tm.do('arch_init'))

        for p in state.data["personas"]:
            p["speaker"] = assign_speaker(p, speaker_pools)

    # Generate ref images for personas
    if a.img_refs > 0 and not a.txtonly and mediagen:
        await mediagen.gen_refs(state.data["personas"], [])
        print(tm.do('refs'))

    # try:
    idx = 1
    cur_vis_prev = []
    while idx <= a.num:
        idx, cur_vis_prev = await gen_debate(idx, a, ai, state, mediagen, cur_vis_prev, speaker_pools)
        print(tm.do(f'-- fragment {idx-1}'))

    # except KeyboardInterrupt:
        # print("\n\nInterrupted by user")
    # except Exception as e:
        # print(f"\n\nError: {e}")

    await state.save()
    print(f"State saved to {logfile}")


if __name__ == '__main__':
    asyncio.run(main())
