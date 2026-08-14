"""The agent loop as a LangGraph graph.

M1 = hand-rolled loop (agent.py). M4 = the same shape, formalized:
StateGraph replaces the for-loop, a conditional edge replaces the
`if not msg.tool_calls` exit check, MemorySaver adds resumable
short-term state across calls — agent.py never had that.

Design call vs the puzzle-book version (wire.py): messages stay as
plain Groq-native dicts/objects, reducer is `operator.add` (list
concat) instead of LangGraph's `add_messages`. `add_messages` upgrades
everything into LangChain message objects, which would force a
translation layer (`to_api`) between graph state and Groq's
tool-calling wire format. Skipping that adapter is the whole payoff —
one less moving part, same data shape agent.py already uses.
"""

import json
import operator
from typing import Annotated, TypedDict

from langgraph.checkpoint.memory import MemorySaver
from langgraph.errors import GraphRecursionError
from langgraph.graph import END, START, StateGraph

from agentic_track.llm import chat
from agentic_track.mcp_bridge import load_tools_from_mcp, mcp_client
from agentic_track.memory import MemoryStore
from agentic_track.prompts import SYSTEM_PROMPT

memory = MemoryStore()


class State(TypedDict):
    messages: Annotated[list, operator.add]
    step: int


def build_graph(tools):
    """Nodes close over `tools` — MCP schema is loaded once per run
    (connection is async, see run_agent), not re-fetched per turn."""

    async def think(state: State):
        msg = chat(state["messages"], tools)
        return {"messages": [msg], "step": state.get("step", 0) + 1}

    async def call_tools(state: State):
        last = state["messages"][-1]
        outputs = []
        for tc in last.tool_calls:
            args = json.loads(tc.function.arguments)
            result = await mcp_client.call_tool(tc.function.name, args)
            outputs.append(
                {
                    "role": "tool",
                    "tool_call_id": tc.id,
                    "content": str(result.data),
                }
            )
        return {"messages": outputs}

    def route(state: State):
        return "act" if state["messages"][-1].tool_calls else "done"

    g = StateGraph(State)
    g.add_node("think", think)
    g.add_node("tools", call_tools)
    g.add_edge(START, "think")
    g.add_conditional_edges("think", route, {"act": "tools", "done": END})
    g.add_edge("tools", "think")
    return g.compile(checkpointer=MemorySaver())


async def run_agent(question, thread_id="default", max_steps=5):
    async with mcp_client:
        tools = await load_tools_from_mcp(mcp_client)
        app = build_graph(tools)

        messages = [{"role": "system", "content": SYSTEM_PROMPT}]
        hits = memory.recall(question, k=3)
        if hits:
            recalled = "\n".join(f"- {h}" for h in hits)
            messages.append(
                {"role": "system", "content": f"Relevant memory:\n{recalled}"}
            )
        messages.append({"role": "user", "content": question})

        cfg = {
            "configurable": {"thread_id": thread_id},
            "recursion_limit": max_steps * 2 + 5,
        }
        try:
            result = await app.ainvoke({"messages": messages, "step": 0}, cfg)
        except GraphRecursionError:
            return "Stopped: hit max steps without final answer."

    answer = result["messages"][-1].content
    memory.save(f"Q: {question}\nA: {answer}")
    return answer


if __name__ == "__main__":
    import asyncio

    print(
        asyncio.run(
            run_agent(
                "What is 47 * 89, and how many words are in "
                "'the agent loop is beating'?"
            )
        )
    )
