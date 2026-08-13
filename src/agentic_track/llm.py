"""The LLM boundary — Groq to think, Gemini to embed."""

from dotenv import load_dotenv
from google import genai
from groq import Groq

load_dotenv()
client = Groq()
gemini = genai.Client()  # GEMINI_API_KEY from .env


def chat(messages, tools):
    resp = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=messages,
        tools=tools,
        tool_choice="auto",
        temperature=0,
    )
    return resp.choices[0].message


def embed(text):
    r = gemini.models.embed_content(model="gemini-embedding-001", contents=text)
    return r.embeddings[0].values
