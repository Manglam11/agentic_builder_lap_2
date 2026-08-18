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

MAX_STEP = 5

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
You are a router. Read the conversation and decide what should happen next.

Reply with exactly one word and nothing else: "researcher" or "writer" or "done".

- Choose "researcher" when the question asks for facts, information, or explanation of something that already exists.
- Choose "writer" when the question asks to create something new — creative or original writing.
- Choose "done" when the conversation already contains a complete answer and no further work is needed.

Do not answer the question. Do not add punctuation or explanation. One word only.
"""
SYSTEM_PROMPT_4 = """
You are a text summarizer. 
Read the user's question and make the summary without loosing any important point.
Your summary should not exceed 100 words.
"""


question_1 = "What is the capital of Japan?"
question_2 = "Write me haiku"
question_3 = "Summairse the causes of 1929 crash"
question_4 = "The economic impact of printing press"
question_5 = "research, then write, then check the France population report"


# def researcher(question: str) -> str:
#     messages = [
#         {"role": "system", "content": SYSTEM_PROMPT_1},
#     ]
#     user_dic = {"role": "user", "content": question}
#     messages.append(user_dic)
#     reply = chat(messages=messages)
#     return reply.content


def researcher(history: list) -> str:
    messages = [{"role": "system", "content": SYSTEM_PROMPT_1}] + history
    reply = chat(messages)
    return reply.content


# def writer(question: str) -> str:
#     messages = [{"role": "system", "content": SYSTEM_PROMPT_2}]
#     user_dic = {"role": "user", "content": question}
#     messages.append(user_dic)
#     reply = chat(messages=messages)
#     return reply.content
def writer(history: list) -> str:
    messages = [{"role": "system", "content": SYSTEM_PROMPT_2}] + history

    reply = chat(messages=messages)
    return reply.content


# def supervise(question: str) -> str:
#     messages = [
#         {"role": "system", "content": SYSTEM_PROMPT_3},
#     ]
#     user_dic = {"role": "user", "content": question}
#     messages.append(user_dic)
#     reply = chat(messages=messages)
#     return reply.content
def supervise(history: list) -> str:
    messages = [{"role": "system", "content": SYSTEM_PROMPT_3}] + history
    reply = chat(messages=messages)
    return reply.content


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


AGENTS = {"researcher": researcher, "writer": writer}


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
        print(f"step: **{step}** and route decision: **{route_decision}**")
        if route_decision == "done":
            return history[-1]["content"]
        else:
            tool_result = orchestrate(route_decision, history)
            history.append({"role": "assistant", "content": tool_result})

            step += 1

    return "stopped: hit max steps"


print(conversation_loop(question_5))
