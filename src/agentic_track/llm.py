"""The LLM boundary — the single Groq call the agent uses to think."""

from groq import Groq
from dotenv import load_dotenv

load_dotenv()

client = Groq()


def chat(messages):
    resp = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=messages,
        temperature=0,
    )
    return resp.choices[0].message.content