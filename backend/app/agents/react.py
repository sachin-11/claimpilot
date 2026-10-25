"""A small ReAct loop as a LangGraph subgraph: model -> tools -> model ... until the model stops
calling tools. Our own tool node (not the prebuilt) so every call goes through the agent's ToolBox:
only its allowed tools, event emission, and a record of every result for citation validation."""

import json
import logging
from collections.abc import Callable
from typing import Annotated, Any, TypedDict

from langchain_core.messages import AIMessage, AnyMessage, BaseMessage, ToolMessage
from langgraph.errors import GraphRecursionError
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages

from app.agents.tools import ToolBox, ToolFailure

logger = logging.getLogger(__name__)

MAX_TOOL_TEXT = 6000


class ReactState(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]


class ToolRecord(TypedDict):
    tool: str
    args: dict[str, Any]
    result: Any


def build_react_graph(  # type: ignore[no-untyped-def]
    llm_with_tools: Any,
    toolbox: ToolBox,
    records: list[ToolRecord],
    render: Callable[[str, Any], str] | None = None,
):
    async def agent(state: ReactState) -> dict[str, list[BaseMessage]]:
        return {"messages": [await llm_with_tools.ainvoke(state["messages"])]}

    async def tools(state: ReactState) -> dict[str, list[BaseMessage]]:
        last = state["messages"][-1]
        assert isinstance(last, AIMessage)
        out: list[BaseMessage] = []
        for call in last.tool_calls:
            try:
                result = await toolbox.call(call["name"], call["args"])
                records.append({"tool": call["name"], "args": call["args"], "result": result})
                text = render(call["name"], result) if render else json.dumps(result, default=str)
            except (ToolFailure, PermissionError) as exc:
                text = f"Tool error: {exc}"  # the model sees it and can adjust
            out.append(ToolMessage(content=text[:MAX_TOOL_TEXT], tool_call_id=call["id"]))
        return {"messages": out}

    def route(state: ReactState) -> str:
        last = state["messages"][-1]
        return "tools" if isinstance(last, AIMessage) and last.tool_calls else END

    graph = StateGraph(ReactState)
    graph.add_node("agent", agent)
    graph.add_node("tools", tools)
    graph.add_edge(START, "agent")
    graph.add_conditional_edges("agent", route, {"tools": "tools", END: END})
    graph.add_edge("tools", "agent")
    return graph.compile()


async def run_react(
    llm_with_tools: Any,
    toolbox: ToolBox,
    messages: list[BaseMessage],
    *,
    max_tool_calls: int,
    render: Callable[[str, Any], str] | None = None,
) -> tuple[list[BaseMessage], list[ToolRecord], bool]:
    """Run the loop with a budget of `max_tool_calls` tool rounds (recursion_limit = 2 steps per
    round). Returns (messages, tool records, budget_exhausted). Running out of budget is not an
    error: the caller still gets everything gathered so far."""
    records: list[ToolRecord] = []
    graph = build_react_graph(llm_with_tools, toolbox, records, render)
    last_values: list[BaseMessage] = list(messages)
    exhausted = False
    try:
        async for values in graph.astream(
            {"messages": messages},
            config={"recursion_limit": 2 * max_tool_calls + 2},
            stream_mode="values",
        ):
            last_values = values["messages"]
    except GraphRecursionError:
        exhausted = True
        logger.warning("react_budget_exhausted", extra={"agent": toolbox.agent})
        # a trailing AI message with unanswered tool calls would be rejected by the next model call
        while last_values and isinstance(last_values[-1], AIMessage) and last_values[-1].tool_calls:
            last_values = last_values[:-1]
    return last_values, records, exhausted
