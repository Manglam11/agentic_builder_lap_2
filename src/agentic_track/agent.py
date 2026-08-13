"""The agent loop — native tool-calling over MCP, now state-threaded with memory.

M2 gave us tool-calling. M3 wraps it: one AgentState bag flows through the loop,
long-term memory is recalled before the model thinks and the answer saved after.
"""

import asyncio
import json

from agentic_track.llm import chat
from agentic_track.mcp_bridge import load_tools_from_mcp, mcp_client
from agentic_track.memory import MemoryStore
from agentic_track.state import new_state

# Long-term memory persists across run_agent calls within one process.
# Restart-proof persistence is the Module 3 project (real vector DB).
memory = MemoryStore()


def _inject_memory(state, question):
    """Recall relevant memories and place them in front of the model."""
    hits = memory.recall(question, k=3)
    state["memory"] = hits
    if hits:
        recalled = "\n".join(f"- {h}" for h in hits)
        state["messages"].insert(
            1, {"role": "system", "content": f"Relevant memory:\n{recalled}"}
        )
    return state


async def step(state, tools):
    """One tool-calling turn: state in -> state out."""
    msg = chat(state["messages"], tools)
    state["messages"].append(msg)
    state["step"] += 1

    if not msg.tool_calls:  # clean exit — the model gave a final answer
        state["done"] = True
        return state

    for tc in msg.tool_calls:
        args = json.loads(tc.function.arguments)
        result = await mcp_client.call_tool(tc.function.name, args)
        state["messages"].append(
            {
                "role": "tool",
                "tool_call_id": tc.id,
                "content": str(result.data),
            }
        )
    return state


async def run_agent(question, max_steps=5):
    state = new_state(question)
    async with mcp_client:
        tools = await load_tools_from_mcp(mcp_client)
        _inject_memory(state, question)
        for _ in range(max_steps):
            state = await step(state, tools)
            if state["done"]:
                answer = state["messages"][-1].content
                memory.save(f"Q: {question}\nA: {answer}")  # remember for next time
                return answer
    return "Stopped: hit max steps without final answer."


if __name__ == "__main__":
    print(
        asyncio.run(
            run_agent(
                "What is 47 * 89, and how many words are in 'the agent loop is beating'?"
            )
        )
    )
