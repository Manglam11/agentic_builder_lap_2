"""The LLM boundary — the single Groq call the agent uses to think."""

from dotenv import load_dotenv
from groq import Groq

load_dotenv()
client = Groq()


def chat(messages, tools):
    resp = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=messages,
        tools=tools,
        tool_choice="auto",
        temperature=0,
    )
    return resp.choices[0].message
