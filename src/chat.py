
import os
import sys
import json
import asyncio
import shutil
import argparse
from itertools import cycle

from base import MediaGen, StoryState, DurationError, load_docs, base_args
from util import Tm, filter_items, fnexist, rand_pick, load_args, file_list, save_cfg

def get_args(parser=None):
    if parser is None: parser = argparse.ArgumentParser(conflict_handler = 'resolve')
    parser.add_argument('-t',  '--in_txt',  default=None, help='Text string or file to process')
    parser.add_argument(       '--maxmsg',  default=6, type=int, help='Amount of recent fragments for context')
    parser.add_argument(       '--updmsg',  default=8, type=int, help='Update personas every N fragments')
    parser.add_argument('-n',  '--num',     default=999, type=int, help='Number of fragments to generate')
    parser.add_argument('-np', '--num_personas', default=3, type=int, help='Number of debate personas')
    # overwrite default
    parser.add_argument('-insd', '--ins_dir', default='data/prompts/chat', help='Instructions directory')
    return parser.parse_args()

start_prompt = 'The world is obscure and incomprehensible. Discuss this topic.'

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
    'writ_vis': {'desc': "Generate visual description for the fragment"},
    'arch_upd': {'desc': "Update persona states based on debate progress", 'tools': ['search']}
}

async def gen_debate(idx, a, ai, state, mediagen=None, cur_vis_prev=[]):
    """Generate single debate fragment with voice and visual"""
    tm = Tm()
    do_media = not a.txtonly and mediagen is not None

    # track last persona for non-repeat selection
    last_persona_name = state.data["fragments"][-1].get("persona_name") if state.data["fragments"] else None
    last_persona = next((p for p in state.data["personas"] if p["name"] == last_persona_name), None)

    # get or gen fragment
    cur_frag = filter_items(state.data["fragments"], fragment_number=idx)
    if not cur_frag:
        persona = rand_pick(state.data["personas"], last_persona)
        recent_frags = state.data["fragments"][-a.maxmsg:] if state.data["fragments"] else [] # context = last K thoughts
        # recent_frags = state.data["voices"][-a.maxmsg:] if state.data["voices"] else [] # context = last K sayings
        cur_input = {
            "global_settings": state.data["global_settings"],
            "fragment_number": idx,
            "persona": persona,
            "recent_fragments": recent_frags
        }
        cur_frag = await ai.call_agent(cur_input, 'writ-frag', 'fragments')
    else:
        persona_name = cur_frag[0].get("persona_name") # from existing fragment
        persona = next((p for p in state.data["personas"] if p["name"] == persona_name), state.data["personas"][0])
    cur_frag = cur_frag[0]
    if a.verbose: print(tm.do('frag %d' % idx))

    speakr = persona.get("speaker", "p230")

    # Media generation
    fmask = os.path.join(a.out_dir, f"{idx:03d}")
    fname = fmask + f"-{speakr}"
    if not fnexist(fmask, 'mp4'):

        # voice-over
        voc_result = None
        for voc_attempt in range(a.max_tries):
            cur_voc = filter_items(state.data["voices"], fragment_number=idx)
            if voc_attempt > 0 or not cur_voc:
                cur_input = {"fragment": cur_frag, "persona": persona, "fragment_number": idx}
                cur_voc = await ai.call_agent(cur_input, 'writ-voc', 'voices')
            cur_voc = cur_voc[0]
            if a.verbose: print(tm.do('voice'))
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
        cur_vis = filter_items(state.data["visuals"], fragment_number=idx)
        if not cur_vis:
            cur_input = {
                "global_settings": state.data["global_settings"],
                "fragment_number": idx,
                "fragment": cur_frag,
                "persona": persona
            }
            cur_vis = await ai.call_agent(cur_input, 'writ-vis', 'visuals')
        cur_vis = cur_vis[0]
        if a.verbose: print(tm.do('previs'))
        # video
        if do_media and voc_result and voc_result['status'] == 'success':
            prompts = cur_vis['content']
            scene_data = {"global_actors": [persona]}  # for ref image lookup
            vis_result = await mediagen.gen_visual(fname, prompts, scene_data=scene_data)
            if a.verbose: print(tm.do('vis'))
            cur_vis_prev = [cur_vis['content']]

    # Periodic persona update
    if idx > 0 and idx % a.updmsg == 0:
        upd_input = {
            "global_settings": state.data["global_settings"],
            "personas": state.data["personas"],
            "recent_fragments": state.data["fragments"][-a.updmsg:]
        }
        result = await ai.call_agent(upd_input, 'arch-upd', save=False)
        # Merge updated personas by name
        if "personas" in result:
            for upd_persona in result["personas"]:
                for i, existing in enumerate(state.data["personas"]):
                    if existing.get("name") == upd_persona.get("name"):
                        upd_persona["speaker"] = existing.get("speaker")
                        state.data["personas"][i] = upd_persona
                        break
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

    # Get initial topic
    if a.in_txt:
        init_topic = open(a.in_txt, 'r', encoding="utf-8").read() if os.path.isfile(a.in_txt) else a.in_txt
    else:
        init_topic = start_prompt

    # Init personas (once)
    if not state.data['personas']:
        init_input = {"initial_topic": init_topic, "num_personas": a.num_personas}
        await ai.call_agent(init_input, 'arch-init', 'personas')
        print(tm.do('arch_init'))

        sp_f = ['f-270','f-306','f-294','f-364','f-245','f-311']
        sp_m = ['m-230','m-317','m-229','m-236','m-262','m-286']
        pools = {'m': cycle(sp_m), 'f': cycle(sp_f)}
        for p in state.data["personas"]:
            p["speaker"] = next(pools.get(p.get("gender","m")[0].lower(), pools['m']))

    # Generate ref images for personas
    if a.img_refs > 0 and not a.txtonly and mediagen:
        await mediagen.gen_refs(state.data["personas"], [])
        print(tm.do('refs'))

    # try:
    idx = 1
    cur_vis_prev = []
    while idx <= a.num:
        idx, cur_vis_prev = await gen_debate(idx, a, ai, state, mediagen, cur_vis_prev)
        print(tm.do(f'-- fragment {idx-1}'))

    # except KeyboardInterrupt:
        # print("\n\nInterrupted by user")
    # except Exception as e:
        # print(f"\n\nError: {e}")

    await state.save()
    print(f"State saved to {logfile}")


if __name__ == '__main__':
    asyncio.run(main())
