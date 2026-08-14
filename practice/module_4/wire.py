import re
from typing import Annotated, TypedDict

from dotenv import load_dotenv
from groq import Groq
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages

load_dotenv()

client = Groq()

SYSTEM_PROMPT = """You are a reasoning agent. You can use these tools:
- calculator[expression]: evaluates a math expression
- word_count[text]: counts the words in a piece of text
Always follow this exact format, one step at a time:
Thought: <your reasoning about what to do next>
Action: <tool_name>[<input>]
Then STOP and wait. You will be given:
Observation: <the tool's result>
Repeat Thought/Action as needed. When you know the
final answer, respond with exactly:
Thought: <your reasoning>
Final Answer: <the answer>
"""


class State(TypedDict):
    messages: Annotated[list, add_messages]
    steps: int


def chat(messages):
    resp = client.chat.completions.create(
        model="llama-3.3-70b-versatile", messages=messages, temperature=0
    )
    return resp.choices[0].message.content


def calculator(expression):
    return str(eval(expression))


def word_count(text):
    return str(len(text.split()))


def parse_action(text: str):
    match = re.search(r"Action:\s*(\w+)\[(.*)\]", text)
    if not match:
        return None, None
    return match.group(1), match.group(2)


TOOLS = {"calculator": calculator, "word_count": word_count}


def to_api(messages):
    role_map = {"human": "user", "ai": "assistant", "system": "system"}
    out = []
    for m in messages:
        api_rol = role_map[m.type]
        out.append({"role": api_rol, "content": m.content})
    return out


def think(state: State):
    resp = chat(to_api(state["messages"]))
    msg = {"role": "assistant", "content": resp}
    return {"steps": state.get("steps", 0) + 1, "messages": [msg]}


def route(state: State):
    final = state["messages"][-1].content
    return "done" if "Final Answer:" in final else "act"


def tools(state: State):
    final = state["messages"][-1].content
    tool_name, tool_input = parse_action(final)
    if tool_name not in TOOLS:
        obs = f"Error: no tool named {tool_name}"
        return {"messages": [{"role": "user", "content": obs}]}
    tool_call = TOOLS[tool_name]
    tool_op = tool_call(tool_input)
    return {"messages": [{"role": "user", "content": f"Observation: {tool_op}"}]}


def last(result):
    return result["messages"][-1].content


g = StateGraph(State)
g.add_node("think", think)
g.add_node("tools", tools)

g.add_edge(START, "think")
g.add_conditional_edges("think", route, {"act": "tools", "done": END})
g.add_edge("tools", "think")

app = g.compile(checkpointer=MemorySaver())
cfg1 = {"configurable": {"thread_id": "chat-1"}}
# result = app.invoke(
#     {
#         "messages": [
#             {"role": "system", "content": SYSTEM_PROMPT},
#             {"role": "user", "content": "What is 47*89?"},
#         ],
#         "steps": 0,
#     },
#     cfg1,
# )

r1 = app.invoke(
    {
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": "My name is Manglam"},
        ],
        "steps": 0,
    },
    cfg1,
)

r2 = app.invoke(
    {
        "messages": [
            {"role": "user", "content": "what is my name?"},
        ]
    },
    cfg1,
)

print("chat-1", last(r2))

cfg2 = {"configurable": {"thread_id": "chat-2"}}

r3 = app.invoke(
    {
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": "What is my name?"},
        ]
    },
    cfg2,
)

print("chat-2", last(r3))

# for m in result["messages"]:
#     print(f"{m.type}:{m.content}")
