"""The agent loop — native tool-calling over MCP.

Same ReAct control loop as Module 1, but the regex parser is gone and the
tools live behind an MCP server instead of a local dict.
"""

import asyncio
import json

from agentic_track.llm import chat
from agentic_track.mcp_bridge import load_tools_from_mcp, mcp_client
from agentic_track.prompts import SYSTEM_PROMPT


async def run_agent(question, max_steps=5):
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": question},
    ]
    async with mcp_client:
        tools = await load_tools_from_mcp(mcp_client)
        for step in range(max_steps):
            msg = chat(messages, tools)
            if not msg.tool_calls:
                return msg.content
            messages.append(msg)
            for tc in msg.tool_calls:
                args = json.loads(tc.function.arguments)
                result = await mcp_client.call_tool(tc.function.name, args)
                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": tc.id,
                        "content": str(result.data),
                    }
                )
    return "Stopped: hit max steps without final answer."


if __name__ == "__main__":
    print(
        asyncio.run(
            run_agent(
                "What is 47 * 89, and how many words are in 'the agent loop is beating'?"
            )
        )
    )
