import json
import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, TypedDict

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from langgraph.graph import END, StateGraph

from src.prompt_temp import react_system_header_str


@dataclass(frozen=True)
class ToolSpec:
    name: str
    description: str
    fn: Any


class LangGraphAgentState(TypedDict):
    messages: List[BaseMessage]
    tool_results: List[str]
    steps: List[Dict[str, str]]
    final_answer: Optional[str]
    pending_action: Optional[Dict[str, str]]
    latest_response: Optional[str]


ACTION_RE = re.compile(r"Action:\s*([A-Za-z0-9_]+)")
INPUT_RE = re.compile(r"Action Input:\s*(\{.*\})", re.S)
THOUGHT_RE = re.compile(r"Thought:\s*(.*)")
ANSWER_RE = re.compile(r"Answer:\s*(.*)", re.S)


def _build_tool_descriptions(tools: List[ToolSpec]) -> str:
    return "\n".join(f"{tool.name}: {tool.description}" for tool in tools)


def _build_system_prompt(tools: List[ToolSpec]) -> str:
    tool_desc = _build_tool_descriptions(tools)
    tool_names = ", ".join(tool.name for tool in tools)
    return react_system_header_str.format(tool_desc=tool_desc, tool_names=tool_names)


def _parse_react_message(content: str) -> Dict[str, Optional[str]]:
    action_match = ACTION_RE.search(content)
    input_match = INPUT_RE.search(content)
    thought_match = THOUGHT_RE.search(content)
    answer_match = ANSWER_RE.search(content)

    return {
        "thought": thought_match.group(1).strip() if thought_match else None,
        "action": action_match.group(1).strip() if action_match else None,
        "action_input": input_match.group(1).strip() if input_match else None,
        "answer": answer_match.group(1).strip() if answer_match else None,
    }


def _safe_json_args(args: Optional[str]) -> Dict[str, Any]:
    if not args:
        return {}
    try:
        return json.loads(args)
    except json.JSONDecodeError:
        return {}


def _llm_node(llm, system_prompt: str):
    def _run(state: LangGraphAgentState) -> Dict[str, Any]:
        messages = state["messages"]
        if not messages or not isinstance(messages[0], SystemMessage):
            messages = [SystemMessage(content=system_prompt)] + messages
        response = llm.invoke(messages)
        return {
            "messages": messages + [response],
            "latest_response": response.content,
        }

    return _run


def _decide_node(state: LangGraphAgentState) -> Dict[str, Any]:
    content = state["messages"][-1].content
    parsed = _parse_react_message(content)
    steps = list(state.get("steps", []))
    updates: Dict[str, Any] = {}

    step: Dict[str, str] = {}
    if parsed["thought"]:
        step["Thought"] = parsed["thought"]
    if parsed["action"]:
        step["Action"] = parsed["action"]
    if parsed["action_input"]:
        step["Action Input"] = parsed["action_input"]
    if parsed["answer"]:
        step["Answer"] = parsed["answer"]

    if step:
        steps.append(step)
        updates["steps"] = steps

    if parsed["action"]:
        updates["pending_action"] = {
            "tool": parsed["action"],
            "args": parsed["action_input"] or "{}",
        }
        return updates

    if parsed["answer"]:
        updates["final_answer"] = f"Answer: {parsed['answer']}"
        return updates

    updates["final_answer"] = content
    return updates


def _tool_node(tool_lookup: Dict[str, ToolSpec]):
    def _run(state: LangGraphAgentState) -> Dict[str, Any]:
        pending = state.get("pending_action") or {}
        tool_name = pending.get("tool")
        args = _safe_json_args(pending.get("args"))
        tool = tool_lookup.get(tool_name)

        if tool is None:
            result = f"Tool '{tool_name}' not found."
        else:
            result = tool.fn(**args)

        observation = f"Observation: {result}"
        messages = list(state["messages"]) + [HumanMessage(content=observation)]
        steps = list(state.get("steps", []))
        if steps:
            steps[-1]["Observation"] = str(result)

        return {
            "messages": messages,
            "tool_results": list(state.get("tool_results", [])) + [str(result)],
            "pending_action": None,
            "steps": steps,
        }

    return _run


def _final_node(state: LangGraphAgentState) -> Dict[str, Any]:
    if not state.get("final_answer"):
        return {"final_answer": state.get("latest_response")}
    return {}


def build_langgraph_agent(llm, tools: List[ToolSpec]) -> StateGraph:
    system_prompt = _build_system_prompt(tools)
    tool_lookup = {tool.name: tool for tool in tools}

    graph = StateGraph(LangGraphAgentState)
    graph.add_node("llm", _llm_node(llm, system_prompt))
    graph.add_node("decide", _decide_node)
    graph.add_node("tools", _tool_node(tool_lookup))
    graph.add_node("final", _final_node)

    graph.set_entry_point("llm")
    graph.add_edge("llm", "decide")

    def _route(state: LangGraphAgentState) -> str:
        if state.get("pending_action"):
            return "tools"
        if state.get("final_answer"):
            return "final"
        return "final"

    graph.add_conditional_edges(
        "decide",
        _route,
        {
            "tools": "tools",
            "final": "final",
        },
    )
    graph.add_edge("tools", "llm")
    graph.add_edge("final", END)

    return graph


def build_initial_state(query: str, tools: List[ToolSpec]) -> LangGraphAgentState:
    system_prompt = _build_system_prompt(tools)
    return LangGraphAgentState(
        messages=[SystemMessage(content=system_prompt), HumanMessage(content=query)],
        tool_results=[],
        steps=[],
        final_answer=None,
        pending_action=None,
        latest_response=None,
    )
