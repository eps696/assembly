
import json
import asyncio
import uuid
from typing import Any, Dict, List, Optional

from anthropic import AsyncAnthropic

from base import PromptMan, parse_json
from tools import get_schema, execute as call_tool

class AgentClaude:
    """Orchestrates Claude agents with session and prompt management"""

    def __init__(self, state, args, prompts_dir, defs, db_url=None):
        self.args = args
        self.evals = args.evals
        self.verbose = args.verbose
        
        self.model = getattr(args, 'txt_model', 'claude-sonnet-4-6')
        self.client = AsyncAnthropic()
        
        self.sessions = {}
        self.persess = db_url is not None
        
        self.state = state
        self.prompter = PromptMan(prompts_dir)
        
        self.agents = self._create_agents(defs)

    def _create_agents(self, defs):
        """Create agent configurations from definitions"""
        agents = {}
        self.instructions = {} # instructions for evaluation
        all_defs = {**defs, 'eval': {'desc': 'Quality assurance evaluator'}}
        for name, feats in all_defs.items():
            prompt_name = name.replace('_', '-')
            instruction = self.prompter.get_prompt(prompt_name)
            self.instructions[name] = instruction
            tools = [s for t in feats.get('tools', []) if (s := get_schema(t, 'claude'))]
            agents[name] = {'name': name, 'description': feats['desc'], 'instruction': instruction, 'tools': tools}
        return agents

    def _get_session(self, agent_name, fresh=False):
        """Get or create a session for an agent"""
        if fresh or agent_name not in self.sessions:
            self.sessions[agent_name] = []
        return self.sessions[agent_name]

    async def _execute_tool(self, tool_name, tool_input):
        """Execute a tool and return the result"""
        result = await asyncio.get_event_loop().run_in_executor(None, lambda: call_tool(tool_name, tool_input))
        return json.dumps(result)

    async def call_agent(self, inputs, run_id, checkout=None, save=True, evals=0, loose_eval=False, context=None):
        """Call Claude agent with input. evals > 0 enables QA evaluation loop"""
        runner_id = run_id.replace('-', '_')
        if evals == 0: evals = self.evals
        instruction = self.instructions.get(runner_id, '')
        orig_inputs = {k: v for k, v in inputs.items()}
        feedbacks, best, best_score = [], None, -1
        
        for attempt in range(max(1, evals + 1)):
            result = await self._ask(runner_id, inputs, context=context)
            if not evals:
                break
            eval_in = {"instruction": instruction, "output": result}
            if not loose_eval: eval_in = {**eval_in, "inputs": orig_inputs}
            eval_result = await self._ask('eval', eval_in, fresh_session=True, use_thinking=True) or {}
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

    async def _ask(self, runner_id, inputs, checkout=None, fresh_session=False, context=None, use_thinking=False):
        """Make a Claude API call with optional caching and extended thinking"""
        agent = self.agents.get(runner_id)
        if not agent:
            raise ValueError(f"Unknown agent: {runner_id}")
        
        # Build system blocks with cache control
        system_blocks = []
        
        # Static instruction - mark for caching
        instruction_block = {"type": "text", "text": agent['instruction']}
        if getattr(self.args, 'use_cache', False):
            instruction_block["cache_control"] = {"type": "ephemeral"}
        system_blocks.append(instruction_block)
        
        # Dynamic context (if provided)
        if context:
            context_text = f"\n\nCurrent context:\n{json.dumps(context, ensure_ascii=False, indent=2)}"
            system_blocks.append({"type": "text", "text": context_text})
        
        # Build messages
        session = self._get_session(runner_id, fresh=fresh_session)
        user_message = json.dumps(inputs, ensure_ascii=False)
        messages = session + [{"role": "user", "content": user_message}]
        
        request_kwargs = {
            "model": self.model,
            "max_tokens": 4096,
            "system": system_blocks,
            "messages": messages,
            # "top_p": 0.9,
            "temperature": 1.0
        }
        
        # Add tools if agent has them
        if agent['tools']:
            request_kwargs["tools"] = agent['tools']
        
        # Extended thinking for eval
        if use_thinking and getattr(self.args, 'use_thinking', False):
            request_kwargs["temperature"] = 1  # Required for extended thinking
            request_kwargs["max_tokens"] = 8192
            request_kwargs["thinking"] = {"type": "enabled", "budget_tokens": getattr(self.args, 'thinking_budget', 5000)}
        
        result = None
        while result is None:
            response = await self.client.messages.create(**request_kwargs)

            # Handle tool use if needed
            while response.stop_reason == "tool_use":
                tool_results = []
                assistant_content = response.content

                for block in response.content:
                    if block.type == "tool_use":
                        tool_result = await self._execute_tool(block.name, block.input)
                        tool_results.append({
                            "type": "tool_result",
                            "tool_use_id": block.id,
                            "content": tool_result
                        })

                messages = messages + [
                    {"role": "assistant", "content": assistant_content},
                    {"role": "user", "content": tool_results}
                ]
                request_kwargs["messages"] = messages
                response = await self.client.messages.create(**request_kwargs)

            # Extract text from response (skip thinking blocks)
            output = ""
            for block in response.content:
                if hasattr(block, 'text'):
                    output += block.text

            try:
                result = parse_json(output)
                if checkout: result[checkout][0].keys()  # validate
                if self.verbose: print('  ', str(result)[:180])
            except:
                print('!! FAIL:', output[:500])
                result = None

            if result and self.persess and not fresh_session:
                session.append({"role": "user", "content": user_message})
                session.append({"role": "assistant", "content": output})

        return result
