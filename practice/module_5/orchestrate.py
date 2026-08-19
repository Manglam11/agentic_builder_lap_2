import operator
from typing import Annotated, TypedDict

from langgraph.graph import END, START, StateGraph

from agentic_track.graph import build_graph
from agentic_track.llm import chat

question_a = "Find me the population of France"
question_b = "Write the poem about ocean"
question_c = "Write a report on France's population"


def route(question: str) -> str:
    question = question.lower()
    if "find" in question or "population" in question:
        return "researcher"
    elif "write" in question or "poem" in question:
        return "writer"


# print(route(question_a))
# print(route(question_b))
# print(route(question_c))

MAX_STEP = 2

SYSTEM_PROMPT_1 = """
You are a helpfull assistant.
You answer facutal lookup concisely.
But if asked to being creative refused.
"""
SYSTEM_PROMPT_2 = """
You are a helpfull assistant.
You write creative or original content"
But if asked to being factual refused.
"""
SYSTEM_PROMPT_3 = """
You are a router coordinating a multi-step task.

First, identify every step the user's original request asks for.
Then look at the conversation for [agent] labels showing which steps already ran.

Reply with exactly one word:
- "researcher" if research is still needed and hasn't run.
- "writer" if writing is still needed and hasn't run.
- "done" ONLY when every step the user asked for has a matching [agent] label in the conversation.

One word. No punctuation. Do not answer the question yourself.
"""
SYSTEM_PROMPT_4 = """
You are a text summarizer. 
Read the user's question and make the summary without loosing any important point.
Your summary should not exceed 100 words.
"""
SYSTEM_PROMPT_5 = """
You are a router coordinating a multi-step task.

First, identify every step the user's original request asks for.
Then look at the conversation for [agent] labels showing which steps already ran.

Reply with exactly one word:
- "research_team" if research is still needed and hasn't run.
- "critic" if review is still needed and hasn't run.
- "done" ONLY when every step the user asked for has a matching [agent] label in the conversation.

One word. No punctuation. Do not answer the question yourself.
"""

SYSTEM_PROMPT_6 = """
You are a critic. You review a report that another agent has written.

Read the most recent report in the conversation. Check it for right and wrong.
Give a short critique: what is correct and what could be more accurate.

Keep it under 100 words. Do not rewrite the report yourself.
"""

question_1 = "What is the capital of Japan?"
question_2 = "Write me haiku"
question_3 = "Summairse the causes of 1929 crash"
question_4 = "The economic impact of printing press"
# question_5 = "research, then write, then check the France population report and provide me summary."
question_5 = "Research France's population, then write a short report on it."


# def researcher(question: str) -> str:
#     messages = [
#         {"role": "system", "content": SYSTEM_PROMPT_1},
#     ]
#     user_dic = {"role": "user", "content": question}
#     messages.append(user_dic)
#     reply = chat(messages=messages)
#     return reply.content


# def researcher(history: list) -> str:
#     messages = [{"role": "system", "content": SYSTEM_PROMPT_1}] + history
#     reply = chat(messages)
#     return reply.content


# def writer(question: str) -> str:
#     messages = [{"role": "system", "content": SYSTEM_PROMPT_2}]
#     user_dic = {"role": "user", "content": question}
#     messages.append(user_dic)
#     reply = chat(messages=messages)
#     return reply.content
# def writer(history: list) -> str:
#     messages = [{"role": "system", "content": SYSTEM_PROMPT_2}] + history

#     reply = chat(messages=messages)
#     return reply.content


# def supervise(question: str) -> str:
#     messages = [
#         {"role": "system", "content": SYSTEM_PROMPT_3},
#     ]
#     user_dic = {"role": "user", "content": question}
#     messages.append(user_dic)
#     reply = chat(messages=messages)
#     return reply.content
# def supervise(history: list) -> str:
#     messages = [{"role": "system", "content": SYSTEM_PROMPT_3}] + history
#     reply = chat(messages=messages)
#     return reply.content


# def orchestrate(text: str, question: str):
#     if text not in AGENTS:
#         return "No tool matched, sorry!!"
#     else:
#         answer = AGENTS[text](question)
#         return answer
def orchestrate(text: str, history: list):
    if text not in AGENTS:
        return "No tool matched, sorry!!"
    else:
        answer = AGENTS[text](history)
        return answer


def summarize(question: str):
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT_4},
    ]
    user_dic = {"role": "user", "content": question}
    messages.append(user_dic)
    reply = chat(messages=messages)
    return reply.content


# print(researcher(question_1))
# print(researcher(question_2))

# print(supervise(question_3))


# tool_name = supervise(question_4)
# full_op = orchestrate(tool_name, question_4)
# summarize_op = summarize(full_op)
# writer_call = writer(summarize_op)
# print(writer_call)


def research_then_write(question: str) -> str:
    findings = researcher(question)  # or route via supervise+orchestrate
    brief = summarize(findings)
    return writer(brief)


# print(orchestrate(tool_name, question_4))


def conversation_loop(question: str):
    step = 0
    history = [{"role": "user", "content": question}]
    while step < MAX_STEP:
        route_decision = supervise(history)
        print(f"step: **{step}**  route decision: **{route_decision}**")
        if route_decision == "done":
            return history[-1]["content"]
        else:
            tool_result = orchestrate(route_decision, history)
            history.append(
                {"role": "assistant", "content": f"[{route_decision}]  {tool_result}"}
            )

            step += 1

    return "stopped: hit max steps"


# print(conversation_loop(question_5))


class SupervisorState(TypedDict):
    messages: Annotated[list, operator.add]
    next: str
    step: int


def researcher(state: SupervisorState) -> dict:
    messages = [{"role": "system", "content": SYSTEM_PROMPT_1}] + state["messages"]
    reply = chat(messages)
    return {
        "messages": [{"role": "assistant", "content": f"[researcher] {reply.content}"}]
    }


def writer(state: SupervisorState) -> dict:
    messages = [{"role": "system", "content": SYSTEM_PROMPT_2}] + state["messages"]
    reply = chat(messages)
    return {"messages": [{"role": "assistant", "content": f"[writer] {reply.content}"}]}


def supervise(state: SupervisorState) -> dict:
    messages = [{"role": "system", "content": SYSTEM_PROMPT_3}] + state["messages"]
    reply = chat(messages)
    decision = reply.content.strip()
    print(f"[supervisor decided]: {repr(decision)}")
    return {"next": decision}


def route(state: SupervisorState) -> str:
    return state["next"]


AGENTS = {"researcher": researcher, "writer": writer}


g = StateGraph(SupervisorState)
g.add_node("supervise", supervise)
g.add_node("researcher", researcher)
g.add_node("writer", writer)

g.add_edge(START, "supervise")
g.add_conditional_edges(
    "supervise", route, {"researcher": "researcher", "writer": "writer", "done": END}
)
g.add_edge("researcher", "supervise")
g.add_edge("writer", "supervise")

research_team = g.compile()


# result = app.invoke(
#     {
#         "messages": [
#             {
#                 "role": "user",
#                 "content": "Research the economic impact of the printing press, then write a short report on it.",
#             }
#         ]
#     },
#     {"recursion_limit": 6},
# )
def research_team_node(state: SupervisorState) -> dict:
    sub_result = research_team.invoke(
        {"messages": state["messages"]},
        {"recursion_limit": 6},
    )
    text: str = sub_result["messages"][-1]["content"]
    clean = text.replace("[writer] ", "").replace("[researcher] ", "")
    return {"messages": [{"role": "assistant", "content": f"[research_team] {clean}"}]}


def critic(state: SupervisorState):
    messages = [{"role": "system", "content": SYSTEM_PROMPT_6}] + state["messages"]
    reply = chat(messages=messages)
    return {"messages": [{"role": "assistant", "content": f"[critic] {reply.content}"}]}


# print(result["messages"][-1])
# print(result["messages"][-1]["content"])


def parent_supervise(state: SupervisorState):
    messages = [{"role": "system", "content": SYSTEM_PROMPT_5}] + state["messages"]
    reply = chat(messages)
    decision = reply.content.strip()
    print(f"[parent decided]: {repr(decision)}")
    return {"next": decision}


parent = StateGraph(SupervisorState)
parent.add_node("parent_supervise", parent_supervise)
parent.add_node("research_team", research_team_node)
parent.add_node("critic", critic)

parent.add_edge(START, "parent_supervise")
parent.add_conditional_edges(
    "parent_supervise",
    route,
    {"research_team": "research_team", "critic": "critic", "done": END},
)
parent.add_edge("research_team", "parent_supervise")
parent.add_edge("critic", "parent_supervise")

app = parent.compile()
result = app.invoke(
    {
        "messages": [
            {
                "role": "user",
                "content": "Research the printing press's economic impact, write a short report, then critique it.",
            }
        ]
    },
    {"recursion_limit": 8},
)

print(result["messages"][-1]["content"])
