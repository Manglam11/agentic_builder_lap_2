import asyncio
import json

from dotenv import load_dotenv
from fastmcp import Client
from groq import Groq

load_dotenv()

groq = Groq()
mcp_client = Client("weather_server.py")

tools = [
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Get the current weather for a given city.",
            "parameters": {
                "type": "object",
                "properties": {
                    "city": {"type": "string", "description": "A city name, eg Mumbai"}
                },
                "required": ["city"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "word_count",
            "description": "Count the number of words in a piece of text.",
            "parameters": {
                "type": "object",
                "properties": {
                    "text": {
                        "type": "string",
                        "description": "Any string, like 'hello world'",
                    }
                },
                "required": ["text"],
            },
        },
    },
]


async def run_agent(question, max_steps=5):
    messages = [{"role": "user", "content": question}]
    async with mcp_client:
        for _ in range(max_steps):
            resp = groq.chat.completions.create(
                model="llama-3.3-70b-versatile",
                messages=messages,
                tools=tools,
                tool_choice="auto",
            )
            msg = resp.choices[0].message
            if not msg.tool_calls:
                return msg.content
            messages.append(msg)
            for tc in msg.tool_calls:
                args = json.loads(tc.function.arguments)
                result = await mcp_client.call_tool(tc.function.name, args)
                messages.append(
                    {"role": "tool", "tool_call_id": tc.id, "content": str(result.data)}
                )
    return "Stopped: hit max steps."


print(asyncio.run(run_agent("What is the weather in Delhi")))
print(asyncio.run(run_agent("How many words in 'the agent loop is beating'")))
