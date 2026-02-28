
import json
import asyncio
import uuid

from google.adk import Agent as LlmAgent
from google.adk.runners import Runner
from google.adk.apps.app import App, EventsCompactionConfig, ResumabilityConfig
from google.adk.sessions import InMemorySessionService, DatabaseSessionService
from google.adk.tools import google_search
from google.adk.models.google_llm import Gemini
from google.genai import types

from base import PromptMan, parse_json

class AgentADK:
    """ADK agents with session and prompt management"""

    def __init__(self, state, args, prompts_dir, defs, db_url=None):
        self.args = args
        self.evals = args.evals
        self.verbose = args.verbose

        if db_url:
            self.sess_service = DatabaseSessionService(db_url=db_url)
            self.rdict = {
                'resumability_config': ResumabilityConfig(is_resumable=True),
                'events_compaction_config': EventsCompactionConfig(compaction_interval=5, overlap_size=1)
            }
            self.persess = True
        else:
            self.sess_service = InMemorySessionService()
            self.rdict = {}
            self.persess = False

        self.state = state
        self.toolset = {'search': google_search}
        self.prompter = PromptMan(prompts_dir)

        self.agents = self._create_agents(defs)
        self.runners = self._create_runners()

    def _create_agents(self, defs):
        retry_cfg = types.HttpRetryOptions(attempts=5, exp_base=7, initial_delay=1, http_status_codes=[429,500,503,504])
        self.model = Gemini(model=self.args.txt_model, temperature=1.1, top_p=0.9, retry_options=retry_cfg)
        adict = {'model': self.model, 'output_key': "result"}
        agents = {}
        self.instructions = {}  # Store instructions for evaluation
        all_defs = {**defs, 'eval': {'desc': 'Quality assurance evaluator'}} # eval agent for QA evaluation
        for name, feats in all_defs.items():
            prompt_name = name.replace('_', '-')
            instruction = self.prompter.get_prompt(prompt_name)
            self.instructions[name] = instruction
            tools = [self.toolset[t] for t in feats.get('tools', []) if t in self.toolset]
            agents[name] = LlmAgent(name=name, description=feats['desc'], instruction=instruction, tools=tools, **adict)
        return agents

    def _create_runners(self):
        runners = {}
        for name, agent in self.agents.items():
            app = App(name=name, root_agent=agent, **self.rdict)
            runners[name] = Runner(app=app, session_service=self.sess_service)
        return runners

    async def _get_sess(self, app_name, sess_id, user_id):
        try:
            sess = await self.sess_service.create_session(app_name=app_name, session_id=sess_id, user_id=user_id)
        except:
            sess = await self.sess_service.get_session(app_name=app_name, session_id=sess_id, user_id=user_id)
        return sess

    async def call_agent(self, inputs, run_id, checkout=None, context=None, save=True, evals=0, loose_eval=False):
        """Call ADK agent with input data. evals > 0 enables QA evaluation loop."""
        runner_id = run_id.replace('-', '_')
        if evals == 0: evals = self.evals
        instruction = self.instructions.get(runner_id, '')
        if context: inputs = {**context, **inputs}
        orig_inputs = {k: v for k, v in inputs.items()}
        feedbacks, best, best_score = [], None, -1

        for attempt in range(max(1, evals + 1)):
            result = await self._ask(runner_id, inputs, checkout)
            if not evals: 
                break
            # Evaluate with fresh session to ensure independence
            eval_in = {"instruction": instruction, "output": result}
            if not loose_eval: eval_in = {**eval_in, "inputs": orig_inputs}
            eval_result = await self._ask('eval', eval_in, fresh_session=True) or {}
            ev = eval_result.get('evaluation', eval_result)
            score = ev.get('score', 0)
            if score > best_score: best, best_score = result, score
            if ev.get('status') == 'APPROVED' or attempt >= evals: 
                break
            if fb := ev.get('feedback', ''):
                feedbacks.append(f"[Attempt {attempt + 1}] {fb}")
                inputs = {**orig_inputs, 'previous_feedbacks': '\n'.join(feedbacks)}
                if self.verbose: print(f".. score {score}, {fb}")

        if best is not None: result = best
        if save:
            await self.state.merge_data(result)
            await self.state.save()
        return result[checkout] if checkout else result

    async def _ask(self, runner_id, inputs, checkout=None, fresh_session=False):
        """Runner call, retry until valid JSON. fresh_session=True ensures no memory bias."""
        runner = self.runners[runner_id]
        use_persistent = self.persess and not fresh_session
        sess_id = runner_id if use_persistent else f"sess_{uuid.uuid4().hex[:8]}"
        sess = await self._get_sess(runner.app_name, sess_id, user_id="system")
        content = types.Content(role='user', parts=[types.Part(text=json.dumps(inputs, ensure_ascii=False))])
        result = None
        while result is None:
            output = ""
            async for event in runner.run_async(user_id="system", session_id=sess.id, new_message=content):
                if event.content and event.content.parts:
                    output += ''.join(p.text for p in event.content.parts if p.text)
            try:
                result = parse_json(output)
                if checkout: result[checkout][0].keys()  # validate
                if self.verbose: print('  ', str(result)[:180])
            except:
                print('!! FAIL:', output[:400])
                result = None
        return result

