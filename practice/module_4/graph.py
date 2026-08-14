from typing import Annotated, TypedDict

from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages

old = [{"role": "user", "content": "hi"}]
new = [{"role": "assistant", "content": "hello"}]
# new = ["hello"]


class State(TypedDict):
    messages: Annotated[list, add_messages]
    steps: int


# merged = add_messages(old, new)
# print(merged)


# def fake_chat(messages):
#     return "Thought: I m thinking"


state = {"messages": [{"role": "user", "content": "hi"}], "steps": 0}


# print(think(state))


def node_a(state):
    return {"messages": [{"role": "assistant", "content": "A"}]}


def node_b(state):
    return {"messages": [{"role": "assistant", "content": "B"}]}


# g = StateGraph(State)
# g.add_node("think", think)
# g.add_node("a", node_a)
# g.add_node("b", node_b)

# g.add_edge(START, "a")
# g.add_edge("a", "b")
# g.add_conditional_edges("think", route, {"act": "tools", "done": END})
# g.add_edge("b", END)

# app = g.compile()

# result = app.invoke({"messages": [{"role": "user", "content": "go"}]})
# print(result["messages"])


acting = {"messages": [{"role": "assistant", "content": "Action: calc[2+2]"}]}
ending = {"messages": [{"role": "assistant", "content": "Final answer: 4"}]}


def route(state: State) -> str:
    # last = state["messages"][-1].content
    # return "act" if "Action:" in last else "done"
    return "done" if state["steps"] >= 5 else "act"


# print(route(acting))
# print(route(ending))

_calls = {"n": 0}


# def fake_chat(messages):
#     _calls["n"] += 1
#     if _calls["n"] < 3:
#         return "Action: calc[2+2]"
#     return "Final Answer: 4"
def fake_chat(messages):
    return "Action: calc[2+2]"


def think(state: State) -> dict:
    reply = fake_chat(state["messages"])
    msg = {"role": "assistant", "content": reply}
    return {"steps": state["steps"] + 1, "messages": [msg]}


def tools(state):
    return {"messages": [{"role": "user", "content": "Observation: 4"}]}


g = StateGraph(State)
g.add_node("think", think)
g.add_node("tools", tools)

g.add_edge(START, "think")
g.add_conditional_edges("think", route, {"act": "tools", "done": END})
g.add_edge("tools", "think")

app = g.compile()

result = app.invoke({"messages": [{"role": "user", "content": "go"}], "steps": 0})
# print(result["messages"])
for m in result["messages"]:
    print(f"{m.type}: {m.content}")
