
import json
import asyncio
import uuid
from typing import Any, Dict, List, Optional, Annotated, TypedDict, Sequence
from operator import add

from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, SystemMessage, ToolMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langchain_core.tools import tool, BaseTool, StructuredTool
from langchain_core.runnables import RunnableConfig

try:
    from langchain_openai import ChatOpenAI
    OPENAI_AVAILABLE = True
except ImportError: OPENAI_AVAILABLE = False
try:
    from langchain_anthropic import ChatAnthropic
    ANTHROPIC_AVAILABLE = True
except ImportError: ANTHROPIC_AVAILABLE = False
try:
    from langchain_google_genai import ChatGoogleGenerativeAI
    GOOGLE_AVAILABLE = True
except ImportError: GOOGLE_AVAILABLE = False

from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode
from langgraph.checkpoint.memory import MemorySaver
try:
    from langgraph.checkpoint.postgres import PostgresSaver
    POSTGRES_AVAILABLE = True
except ImportError: POSTGRES_AVAILABLE = False

from base import PromptMan, parse_json
from tools import get_schema, names as tool_names

class AgentState(TypedDict):
    """State for the agent graph"""
    messages: Annotated[Sequence[BaseMessage], add]
    input_data: Dict[str, Any]
    output: Optional[str]
    iteration: int

@tool
def web_search_tavily(query):
# def search(query):
    """Search using Tavily API (if available).
    Args:
        query: The search query string
    Returns:
        JSON string with search results
    """
    try:
        from langchain_tavily import TavilySearch
        tavily = TavilySearch(max_results=3)
        results = tavily.invoke(query)
        return json.dumps({"status": "success", "results": results})
    except Exception as e:
        return json.dumps({"status": "error", "message": str(e)})


class AgentLang:
    """LangChain/LangGraph agent with optional graph-based execution"""

    def __init__(self, state, args, prompts_dir, defs, db_url=None):
        self.args = args
        self.evals = getattr(args, 'evals', 0)
        self.verbose = getattr(args, 'verbose', False)
        self.use_graph = args.use_graph # LangChain message loop (lighter, simpler) or LangGraph StateGraph with checkpointer (persistent sessions, complex workflows)

        self.model_name = getattr(args, 'txt_model', None)
        self.llm = self._create_llm()

        self.state = state
        self.prompter = PromptMan(prompts_dir)
        self.toolset = {n: StructuredTool.from_function(**get_schema(n, 'lang')) for n in tool_names()}
        self.toolset['search'] = web_search_tavily

        self.agents = self._create_agents(defs)

        if self.use_graph:
            # LangGraph mode: checkpointer + compiled graphs
            if db_url and POSTGRES_AVAILABLE:
                self.checkpointer = PostgresSaver.from_conn_string(db_url)
                self.persess = True
            else:
                self.checkpointer = MemorySaver()
                self.persess = db_url is not None
            self.graphs = self._create_graphs()
        else:
            # Plain LangChain mode: in-memory session history
            self.sessions: Dict[str, List[BaseMessage]] = {}
            self.persess = db_url is not None

    def _create_llm(self):
        """Create LLM instance based on available providers and args"""
        model_name = self.model_name
        if ANTHROPIC_AVAILABLE and (model_name is None or 'claude' in str(model_name).lower()):
            return ChatAnthropic(model=model_name or "claude-sonnet-4-6", temperature=1.0, max_tokens=64000)
        if OPENAI_AVAILABLE and (model_name is None or 'gpt' in str(model_name).lower()):
            return ChatOpenAI(model=model_name or "gpt-5-mini", temperature=1.0, max_tokens=16384)
        if GOOGLE_AVAILABLE and (model_name is None or 'gemini' in str(model_name).lower()):
            return ChatGoogleGenerativeAI(model=model_name or "gemini-3-flash-preview", temperature=1.0)
        raise RuntimeError("No LLM provider available. Install langchain-anthropic, langchain-openai, or langchain-google-genai")

    def _create_agents(self, defs):
        """Create agent configurations from definitions"""
        agents = {}
        self.instructions = {}
        all_defs = {**defs, 'eval': {'desc': 'Quality assurance evaluator'}}
        for name, feats in all_defs.items():
            prompt_name = name.replace('_', '-')
            instruction = self.prompter.get_prompt(prompt_name)
            self.instructions[name] = instruction
            tools = [self.toolset[t] for t in feats.get('tools', []) if t in self.toolset]
            llm = self.llm.bind_tools(tools) if tools else self.llm
            agents[name] = {'name': name, 'description': feats['desc'], 'instruction': instruction, 'tools': tools, 'llm': llm}
        return agents

    # --- Shared interface ---

    async def call_agent(self, inputs, run_id, checkout=None, context=None, save=True, evals=0, loose_eval=False):
        """Call agent with input data. evals > 0 enables QA evaluation loop."""
        runner_id = run_id.replace('-', '_')
        if evals == 0: evals = self.evals
        instruction = self.instructions.get(runner_id, '')
        if context: inputs = {**context, **inputs}
        orig_inputs = {k: v for k, v in inputs.items()}
        feedbacks, best, best_score = [], None, -1

        for attempt in range(max(1, evals + 1)):
            result = await self._ask(runner_id, inputs)
            if not evals:
                break
            # Evaluate
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
                if self.verbose: print(f".. score {score}, {fb}")
                inputs = {**orig_inputs, 'previous_feedbacks': '\n'.join(feedbacks)}

        if best is not None: result = best
        if save:
            await self.state.merge_data(result)
            await self.state.save()
        return result[checkout] if checkout else result

    async def _ask(self, runner_id, inputs, checkout=None, fresh_session=False):
        """Dispatch to graph or chain execution"""
        if self.use_graph:
            return await self._ask_graph(runner_id, inputs, checkout, fresh_session)
        else:
            return await self._ask_chain(runner_id, inputs, checkout, fresh_session)

    # --- LangGraph path ---

    def _create_graphs(self):
        """Create LangGraph state graphs for each agent"""
        graphs = {}
        for name, config in self.agents.items():
            graph = self._build_agent_graph(config)
            graphs[name] = graph
        return graphs

    def _build_agent_graph(self, config):
        """Build a LangGraph for a single agent"""
        tools = config['tools']
        instruction = config['instruction']

        llm = self.llm.bind_tools(tools) if tools else self.llm

        async def agent_node(state):
            """Main agent reasoning node"""
            messages = list(state["messages"])
            # Add system message with instruction
            if not any(isinstance(m, SystemMessage) for m in messages):
                messages.insert(0, SystemMessage(content=instruction))
            # Invoke LLM
            response = await llm.ainvoke(messages)
            return {"messages": [response]}

        def should_continue(state):
            """Determine if we should call tools or end"""
            messages = state["messages"]
            last_message = messages[-1]
            # If there are tool calls, route to tools
            if hasattr(last_message, 'tool_calls') and last_message.tool_calls:
                return "tools"
            return "end"

        workflow = StateGraph(AgentState)
        workflow.add_node("agent", agent_node)

        if tools:
            tool_node = ToolNode(tools)
            workflow.add_node("tools", tool_node)
            # Add edges
            workflow.set_entry_point("agent")
            workflow.add_conditional_edges("agent", should_continue, {"tools": "tools", "end": END})
            workflow.add_edge("tools", "agent")
        else:
            workflow.set_entry_point("agent")
            workflow.add_edge("agent", END)

        # Compile with checkpointer
        return workflow.compile(checkpointer=self.checkpointer)

    def _get_thread_id(self, runner_id, fresh=False):
        """Get or create a thread ID for session management"""
        if fresh:
            return f"thread_{uuid.uuid4().hex[:8]}"
        return f"thread_{runner_id}"

    async def _ask_graph(self, runner_id, inputs, checkout=None, fresh_session=False):
        """Run LangGraph agent, retry until valid JSON"""
        graph = self.graphs.get(runner_id)
        if not graph:
            raise ValueError(f"Unknown agent: {runner_id}")

        user_message = json.dumps(inputs, ensure_ascii=False)
        thread_id = self._get_thread_id(runner_id, fresh=fresh_session)

        result = None
        while result is None:
            initial_state: AgentState = {"messages": [HumanMessage(content=user_message)], "input_data": inputs, "output": None, "iteration": 0}

            # Run the graph
            config = RunnableConfig(configurable={"thread_id": thread_id}, recursion_limit=10)
            final_state = await graph.ainvoke(initial_state, config)

            # Extract output from last AI message
            output = ""
            for msg in reversed(final_state["messages"]):
                if isinstance(msg, AIMessage) and msg.content:
                    if isinstance(msg.content, str):
                        output = msg.content
                    elif isinstance(msg.content, list):
                        # Handle structured content
                        output = "".join(block.get("text", "") if isinstance(block, dict) else str(block) for block in msg.content)
                    break
            try:
                result = parse_json(output)
                if checkout: result[checkout][0].keys()
                if self.verbose: print('  ', str(result)[:180])
            except:
                print('!! FAIL:', output[:500])
                result = None

        return result

    # --- LangChain path ---

    def _get_session(self, agent_name, fresh=False):
        """Get or create session history"""
        if fresh or agent_name not in self.sessions:
            self.sessions[agent_name] = []
        return self.sessions[agent_name]

    async def _ask_chain(self, runner_id, inputs, checkout=None, fresh_session=False):
        """Make LangChain call, retry until valid JSON"""
        agent = self.agents.get(runner_id)
        if not agent:
            raise ValueError(f"Unknown agent: {runner_id}")

        history = self._get_session(runner_id, fresh=fresh_session)
        user_message = json.dumps(inputs, ensure_ascii=False)

        result = None
        while result is None:
            messages = [SystemMessage(content=agent['instruction'])] + history + [HumanMessage(content=user_message)]
            response = await agent['llm'].ainvoke(messages)

            # Handle tool calls if any
            tools = agent['tools']
            while hasattr(response, 'tool_calls') and response.tool_calls and tools:
                # Execute tools
                tool_messages = []
                for tool_call in response.tool_calls:
                    tool_name = tool_call['name']
                    tool_args = tool_call['args']
                    # Find and execute tool
                    for t in tools:
                        if t.name == tool_name:
                            tool_result = await asyncio.to_thread(t.invoke, tool_args)
                            tool_messages.append(ToolMessage(content=str(tool_result), tool_call_id=tool_call['id']))
                            break
                # Continue conversation with tool results
                messages = messages + [response, *tool_messages]
                response = await agent['llm'].ainvoke(messages)

            # Extract text content
            if isinstance(response.content, str):
                output = response.content
            elif isinstance(response.content, list):
                output = "".join(block.get("text", "") if isinstance(block, dict) else str(block) for block in response.content)
            else:
                output = str(response.content)

            try:
                result = parse_json(output)
                if checkout: result[checkout][0].keys()
                if self.verbose: print('  ', str(result)[:180])
            except:
                print('!! FAIL:', output[:500])
                result = None

            # Update history if persistent
            if result and self.persess and not fresh_session:
                history.append(HumanMessage(content=user_message))
                history.append(response)

        return result

