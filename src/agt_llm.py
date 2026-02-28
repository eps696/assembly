
import json
import asyncio
import re

from openai import AsyncOpenAI

from base import PromptMan, parse_json
from tools import get_schemas, get_schema, execute as call_tool
from tool_think import print_thoughts # registers sequential_thinking into _tools on import
toolsets = get_schemas('openai') # after all necessary tools are imported!

# from tools import toolset
# from tool_think import toolset_think, sequential_thinking, print_thoughts
# toolsets = []
# [toolsets.append(d) for d in toolset + toolset_think if d not in toolsets]

# Tags for detecting malformed tool calls in text
TOOL_TAGS = ['[TOOL_REQUEST]', '[END_TOOL_REQUEST]', '[TOOL_CALL]', '[END_TOOL_CALL]',
             '[tool_call]', '[/tool_call]', '<tool_call>', '</tool_call>',
             '<|action_start|><|plugin|>', '<|action_end|>', '<|im_start|>', '<|im_end|>',
             '[END_TOOL_RESULT]']
TOOL_PATTERNS = [re.escape(tag).replace(r'\|', r'\|') for tag in TOOL_TAGS]

def set_msg(content, role='user', name=None):
    """Create message dict for LLM"""
    msg = {"role": role, "content": content}
    if name:
        msg['name'] = name
    return msg

class AgentLLM:
    """Orchestrates LLM agents with prompt management"""

    def __init__(self, state, args, prompts_dir):
        self.args = args
        self.evals = args.evals
        self.verbose = args.verbose
        self.state = state
        self.prompter = PromptMan(prompts_dir)
        self.client = LLM(host=args.llm_host)
        self.client.temp = 1.1
        self.client.top_p = 0.9
        self.client.set_model(args.txt_model)

    async def call_agent(self, inputs, prompt_name, checkout=None, context=None, save=True, evals=0, loose_eval=False):
        """Call LLM with prompt. evals > 0 enables QA evaluation loop."""
        if evals == 0: evals = self.evals
        instruction = self.prompter.get_prompt(prompt_name)
        if context: inputs = {**context, **inputs}
        orig_inputs = {k: v for k, v in inputs.items()}
        feedbacks, best, best_score = [], None, -1

        for attempt in range(max(1, evals + 1)):
            result = None
            while result is None: # retry until valid JSON
                result = await self._ask(instruction, inputs, toolsets)
                if result and checkout:
                    try: result[checkout][0].keys()
                    except: result = None
            if not evals:
                break
            # Evaluate
            eval_in = {"instruction": instruction, "output": result}
            if not loose_eval: eval_in = {**eval_in, "inputs": orig_inputs}
            eval_result = await self._ask(self.prompter.get_prompt('eval'), eval_in, [get_schema('count_text', 'openai')]) or {}
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

    async def _ask(self, prompt, inputs, tools=None):
        """Single LLM call, returns parsed JSON or None."""
        self.client.set_system(prompt, tools)
        msgs = [set_msg(json.dumps(inputs, ensure_ascii=False))]
        output = await self.client.ask(msgs, verbose=False)
        try:
            result = parse_json(output.strip())
            if self.verbose: print('  ', str(result)[:180])
            return result
        except:
            print('!! FAIL:', output[:400])
            return None

class LLM:
    """Async LLM client for LMStudio/OpenAI"""

    def __init__(self, model='gpt-4o-mini', host=None):
        api_args = {'base_url': f"http://{host}:1234/v1", 'api_key': "lm-studio"} if host else {}
        self.client = AsyncOpenAI(**api_args)
        self.model = model
        self.temp = 0.8
        self.top_p = 0.9
        self.max_tokens = 16384 # self.model_info.max_tokens
        self.max_tool_calls = 10
        self.sys_msg = {"role": "system", "content": "You are a helpful assistant."}
        self.tool_msg = self.sys_msg.copy()
        self.tools = []
        self.messages = []

    async def get_models(self):
        self.models = [model.id async for model in self.client.models.list()]
        return self.models

    def set_model(self, model):
        self.model = model
        return self.model

    def set_system(self, prompt, tools=None):
        self.sys_msg = {"role": "system", "content": prompt}
        self.tool_msg = {"role": "system", "content": prompt}
        if tools is not None: #  and len(tools) > 0
            self.set_tools(tools)
        post = ". Ok, so here we go, the chat is starting."
        self.tool_msg['content'] += post
        self.sys_msg['content'] += post

    def set_tools(self, tools):
        self.tools = tools
        if not tools: return
        tool_descs = '; '.join([f"**{t['function']['name']}** for {t['function']['description']}" for t in tools])
        tool_prompt = f"{self.sys_msg['content']} You can use ONLY the following tools (once), and must NEVER make up tools: {tool_descs}. " \
                      f"Enclose tool calls as JSON within XML tags: <tool_call>{{\"name\": <function-name>, \"arguments\": <args-json-object>}}</tool_call>. " \
                      f"DO NOT use any other unnecessary markup."
        self.tool_msg = {"role": "system", "content": tool_prompt}

    def add_msg(self, content, role='user'): # user assistant
        self.messages += [{"role": role, "content": content}]

    async def ask(self, messages=None, del_br=True, verbose=False, **kwargs):
        """Async LLM call with tool handling"""
        if messages is None: messages = self.messages
        temp = kwargs.get('temp', self.temp)
        top_p = kwargs.get('top_p', self.top_p)
        gen_args = {'model': self.model, 'temperature': temp, 'top_p': top_p, 'max_tokens': self.max_tokens}
        tool_args = {'tools': self.tools} if self.tools else {}
        sys_msg = self.tool_msg if self.tools else self.sys_msg
        ask_msgs = [sys_msg] + messages

        tool_call_count = 0
        tool_msgs = []
        answer = None

        while tool_call_count < self.max_tool_calls:
            try:
                response = await self.client.chat.completions.create(messages=ask_msgs + tool_msgs, **gen_args, **tool_args)
                rmsg = response.choices[0].message
            except Exception as e:
                print(f" !! API error: {e}")
                return f"Error: {e}"

            if verbose:
                print(f'\n -- response: {str(rmsg.content)[:160] if rmsg.content else "[no content]"}')

            if hasattr(rmsg, 'tool_calls') and rmsg.tool_calls and not rmsg.content: # proper tool calls
                if verbose: print(f'--- direct tool calls: {[tc.function.name for tc in rmsg.tool_calls]}')
                tool_msgs.extend(await self._process_tool_calls(rmsg.tool_calls, verbose))
                tool_call_count += 1

            elif rmsg.content and any(tag in rmsg.content for tag in TOOL_TAGS): # malformed tool calls
                clean_content, extract_calls = _extract_tool_calls(rmsg.content)
                if verbose:
                    print('--- clean_content', clean_content)
                    print('--- extract_calls', [tc.function.name for tc in extract_calls])
                if clean_content.strip():
                    tool_msgs.append({"role": "assistant", "content": clean_content})
                if extract_calls:
                    tool_msgs.extend(await self._process_tool_calls(extract_calls, verbose))
                    tool_call_count += 1
                elif clean_content: # No tool calls actually processed
                    answer = clean_content
                    break

            else: # no tool calls, final answer
                answer = rmsg.content
                break

        if tool_call_count >= self.max_tool_calls and answer is None: # Exceeded tool call limit - get final answer without tools
            print(' !!! tool calls exceeded, final answer')
            response = await self.client.chat.completions.create(messages=[self.sys_msg] + messages + tool_msgs, **gen_args)
            answer = response.choices[0].message.content
            answer, _ = _extract_tool_calls(answer) # remove tool calls

        if isinstance(answer, str):
            answer = _clean_tags(answer, del_br=del_br)
        self.add_msg(answer, 'assistant')
        return answer

    async def _process_tool_calls(self, tool_calls, verbose=False):
        """Execute tool calls and return result messages"""
        ass_msg = {"role": "assistant", "content": "",
            "tool_calls": [{"id": tc.id, "type": tc.type, "function": {"name": tc.function.name, "arguments": tc.function.arguments or ""}} for tc in tool_calls]}
        result_msgs = [ass_msg]

        for tc in tool_calls:
            func_name = tc.function.name
            try:
                func_args = json.loads(tc.function.arguments) if tc.function.arguments.strip() else {}
                result = call_tool(func_name, func_args)

                if isinstance(result, dict) and "status" in result:
                    if verbose: print(f' -tool- {result["status"]}: {str(result.get("content", ""))[:100]}')
                tool_content = json.dumps(result)

                result_msgs.append({"role": "tool", "content": tool_content, "tool_call_id": tc.id})

            except Exception as e:
                error_msg = f"Error in {func_name}: {e}"
                if verbose: print(f' -tool error- {error_msg}')
                result_msgs.append({"role": "tool", "content": json.dumps({"error": error_msg}), "tool_call_id": tc.id})

        return result_msgs

# === Helper functions ===

def _clean_tags(txt, del_br=True):
    """Remove thinking tags and clean text"""
    out = re.sub(r'<think>.*?</think>', '', txt, flags=re.DOTALL).replace('—', ' — ')
    if del_br: out = out.replace('\n', ' ') # clean qwen etc
    return out.strip()

def _fix_braces(text):
    """Fix malformed JSON braces"""
    text = text.strip().replace('\n', '')
    if not text: return text
    while text.startswith('{{') or text.startswith('{ {'):
        text = text[1:].strip()
    while text.endswith('}}') or text.endswith('} }'):
        text = text[:-1].strip()
    open_count, close_count = text.count('{'), text.count('}')
    if close_count > open_count:
        text = text.rstrip('}' * (close_count - open_count))
    elif open_count > close_count:
        text += '}' * (open_count - close_count)
    return text

def _extract_tool_calls(content):
    """Extract tool calls from text and return (clean_content, tool_calls)"""
    extracted = []
    call_id = 1
    opening_tags = {'[TOOL_REQUEST]', '[TOOL_CALL]', '[tool_call]', '<tool_call>', '<|action_start|><|plugin|>', '<|im_start|>'}

    # Find all tag positions
    tag_positions = []
    for pattern in TOOL_PATTERNS:
        for match in re.finditer(pattern, content):
            tag_positions.append({'start': match.start(), 'end': match.end(), 'tag': match.group(0)})
    tag_positions.sort(key=lambda x: x['start'])

    removal_ranges = []
    i = 0
    while i < len(tag_positions):
        tag = tag_positions[i]
        if tag['tag'] in opening_tags:
            region_start = tag['start']
            content_start = tag['end']
            if i + 1 < len(tag_positions):
                next_tag = tag_positions[i + 1]
                tool_content = content[content_start:next_tag['start']]
                if next_tag['tag'] not in opening_tags:
                    region_end = next_tag['end']
                    i += 1
                else:
                    region_end = next_tag['start']
            else:
                tool_content = content[content_start:]
                region_end = len(content)

            removal_ranges.append((region_start, region_end))

            # Try to parse tool call
            inner = _fix_braces(tool_content)
            try:
                data = json.loads(inner)
                if data.get('name'):
                    args = data.get('arguments', {})
                    tc = type('ToolCall', (), {
                        'id': f'text_call_{call_id}', 'type': 'function',
                        'function': type('Function', (), {
                            'name': data['name'],
                            'arguments': json.dumps(args) if isinstance(args, dict) else args
                        })
                    })
                    extracted.append(tc)
                    call_id += 1
            except json.JSONDecodeError:
                if '{' in inner and '}' in inner:
                    try:
                        start_brace = inner.find('{')
                        end_brace = inner.rfind('}') + 1
                        data = json.loads(inner[start_brace:end_brace])
                        if data.get('name'):
                            args = data.get('arguments', {})
                            tc = type('ToolCall', (), {
                                'id': f'text_call_{call_id}', 'type': 'function',
                                'function': type('Function', (), {
                                    'name': data['name'],
                                    'arguments': json.dumps(args) if isinstance(args, dict) else args
                                })
                            })
                            extracted.append(tc)
                            call_id += 1
                    except:
                        pass
        else:
            removal_ranges.append((tag['start'], tag['end']))
        i += 1

    # Remove tool call regions from content
    clean = content
    for start, end in sorted(set(removal_ranges), reverse=True):
        clean = clean[:start] + clean[end:]
    for tag in TOOL_TAGS:
        clean = clean.replace(tag, '')

    return clean.strip(), extracted


async def chat(host='localhost', model=None):
    """Simple interactive chatbot"""
    prompter = PromptMan('data/prompts/chat')
    client = LLM(host=host)

    models = await client.get_models()
    print(f"Available models: {models}")
    if model and model in models:
        client.set_model(model)
    elif models:
        client.set_model(models[0])
    print(f"Using model: {client.model}")

    sys_prompt = prompter.get_prompt('ass-en') or "You are a helpful and honest assistant."
    client.set_system(sys_prompt, toolsets)

    from util import savelog
    i = 0
    print("\nChat ready. Type 'q' to quit.\n")
    while True:
        user_input = input("You: ").strip()
        if user_input.lower() == 'q':
            print("Goodbye!")
            break
        if not user_input: continue

        client.add_msg(user_input, role='user')
        savelog('\n %02d = = = %s:\n - %s \n' % (i, 'User', user_input))
        response = await client.ask(verbose=False)
        savelog('\n %02d = = = %s:\n - %s \n' % (i, 'Model', response))
        print_thoughts()
        i += 1


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('-lh', '--host', default='localhost', help='LLM server host')
    parser.add_argument('-m', '--model', default=None, help='Model name')
    args = parser.parse_args()

    asyncio.run(chat(host=args.host, model=args.model))
