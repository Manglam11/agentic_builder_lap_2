import math
from typing import TypedDict

from dotenv import load_dotenv
from google import genai
from groq import Groq

load_dotenv()

client = Groq()


# history = [{"role": "system", "content": "you are a helpful assistant"}]


# def remember(role, content):
#     history.append({"role": role, "content": content})


# user_turn = "My name is Manglam"
# assistant_turn = "Nice to meet you, Manglam"

# remember("user", user_turn)
# remember("assistant", assistant_turn)

# print(len(history))
# print(history[-1])


def model(messages):
    seen = " ".join(m["content"] for m in messages)
    return "I know your name!" if "Manglam" in seen else "I don't know your name."


question = {"role": "user", "content": "What's my name?"}
earlier = {"role": "user", "content": "My name is Manglam."}

# print(model([question, earlier]))
# history = [{"role": "system", "content": "you are a helpful assistant"}]

# messages = history.append(question)

# response = client.chat.completions.create(
#     model="llama-3.3-70b-versatile",
#     messages=messages,
# )

# reply = response.choices[0].message.content
# print(reply)

# history_list = [
#     {"role": "system", "content": "RULES"},
#     {"role": "user", "content": "m1"},
#     {"role": "assistant", "content": "r1"},
#     {"role": "user", "content": "m2"},
#     {"role": "assistant", "content": "r2"},
#     {"role": "user", "content": "m3"},
#     {"role": "assistant", "content": "r3"},
# ]
# MAX_TURNS = 4


# def trim_history(history, max_turns):
#     system = history[:1]
#     rest = history[1:][-max_turns:]
#     return system + rest


# print(trim_history(history_list, MAX_TURNS))


# client = genai.Client()  # GEMINI_API_KEY lives in your .env


# def embed(text):
#     r = client.models.embed_content(model="gemini-embedding-001", contents=text)
#     return r.embeddings[0].values


# store = []


# def save_memory(text):
#     embed_value = embed(text)
#     store.append({"text": text, "vec": embed_value})


# save_memory("Manglam is learning agentic AI")
# save_memory("Manglam's stack is gemini and groq")

# # print(len(store))
# # print(len(store[0]["vec"]))


# def cosine(a, b):
#     dot = sum(x * y for x, y in zip(a, b))
#     na = math.sqrt(sum(x * x for x in a))
#     nb = math.sqrt(sum(y * y for y in b))
#     return dot / (na * nb)


# def recall(query, k=1):
#     embed_q = embed(query)
#     # save_memory(embed_q)
#     similarity_socre = []
#     for item in store:
#         cosine_similarity = cosine(embed_q, item["vec"])
#         similarity_socre.append((cosine_similarity, item["text"]))

#     # for _ in range(k):
#     #     return similarity_socre[:k]
#     similarity_socre.sort(reverse=True)
#     return similarity_socre[0][1]


# print(recall("What's Manglam studying?", k=1))


SYSTEM_PROMPT = "You are a helpful agent."


class AgentState(TypedDict):
    messages: list
    memory: list
    step: int
    done: bool


state: AgentState = {
    "messages": [{"role": "system", "content": SYSTEM_PROMPT}],
    "memory": [],
    "step": 0,
    "done": False,
}

# print(state["step"])
# print(len(state["messages"]))


def fake_model(messages):
    turns = sum(1 for m in messages if m["role"] == "assistant")
    if turns >= 1:
        return "Final Answer: done."
    return "Thinking... one more step."


def step(state):
    fake_call = fake_model(state["messages"])
    state["messages"].append({"role": "assistant", "content": fake_call})
    state["step"] += 1
    state["done"] = "Final Answer:" in fake_call
    return state


# result = step(state=state)
# print(result["step"])
# print(result["done"])
# print(result["messages"][-1]["content"])

MAX_STEPS = 5


def run_state(state, max_steps):
    for _ in range(max_steps):
        result = step(state)
        if result["done"] == True:
            return state
    return state


state_result = run_state(state, MAX_STEPS)
print(state_result["step"])
print(state_result["messages"][-1]["content"])
