class Usage:
    def __init__(self, p, c):
        self.prompt_tokens = p
        self.completion_tokens = c


class Reply:
    def __init__(self, usage):
        self.usage = usage


reply = Reply(Usage(1200, 350))
reply_b = Reply(None)

INPUT_PRICE = 0.15
OUTPUT_PRICE = 0.60


def call_cost(reply: Usage):
    try:
        pt = reply.usage.prompt_tokens
        ct = reply.usage.completion_tokens
        cost = (pt / 1000) * INPUT_PRICE + (ct / 1000) * OUTPUT_PRICE
        return cost
    except AttributeError as e:
        return f"Attribute Error: {e}"


# print(call_cost(reply))
# print(call_cost(reply_b))

TRACE = [
    {
        "name": "supervise",
        "seconds": 0.4,
        "prompt_tokens": 120,
        "completion_tokens": 15,
    },
    {
        "name": "researcher",
        "seconds": 1.1,
        "prompt_tokens": 300,
        "completion_tokens": 610,
    },
    {
        "name": "supervise",
        "seconds": 0.3,
        "prompt_tokens": 460,
        "completion_tokens": 12,
    },
    {"name": "writer", "seconds": 0.8, "prompt_tokens": 540, "completion_tokens": 740},
    {"name": "supervise", "seconds": 0.3, "prompt_tokens": 700, "completion_tokens": 6},
]


def trace_data(data: list):
    steps = 0
    slowest = data[0]["seconds"]
    slowest_work = data[0]["name"]
    run_cost = 0
    for i in data:
        steps += 1
        if i["seconds"] > slowest:
            slowest = i["seconds"]

            slowest_work = i["name"]

        current_cost = (i["prompt_tokens"] / 1000) * INPUT_PRICE + (
            i["completion_tokens"] / 1000
        ) * OUTPUT_PRICE
        run_cost += current_cost
    return steps, slowest_work, round(run_cost, 2)


# step_count, slowest_task, cost = trace_data(TRACE)
# print(step_count)
# print(slowest_task)
# print(cost)

expected = ["research_team", "critic", "done"]
actual_a = ["research_team", "critic", "done"]  # clean
actual_b = ["research_team", "research_team", "critic", "done"]  # looped once
actual_c = ["research_team", "done"]  # skipped critic
actual_d = ["critic", "research_team", "done"]


# def trajactory_ok(actual, expected):
#     need = 0
#     # if len(expected) > len(actual):
#     #     return False
#     for i in range(len(actual)):
#         for j in range(len(expected)):
#             if actual[i] == expected[j]:
#                 need += 1
#     # print("actual lenght", len(actual))
#     # print("expected length", len(expected))
#     # print("need value", need)
#     # print()
#     return need == len(actual)


def trajectory_ok(actual, expected):
    need = 0
    for item in actual:
        if item == expected[need]:
            need += 1
            if need == len(expected):
                break
    return need == len(expected)


# print(trajectory_ok(expected, actual_a))
# print(trajectory_ok(expected, actual_b))
# print(trajectory_ok(expected, actual_c))
# print(trajectory_ok(expected, actual_d))


CALLS = [
    {"tool": "search", "ok": True},
    {"tool": "search", "ok": False},
    {"tool": "calc", "ok": True},
    {"tool": "calc", "ok": True},
]
CALLS_EMPTY = []


def success_rate(calls: list):
    total_call = len(calls)
    if total_call == 0:
        return 0.0
    true_call = 0
    for c in calls:
        if c["ok"]:
            true_call += 1
    success_pct = (true_call / total_call) * 100
    return round(success_pct, 2)


# print(success_rate(CALLS))
# print(success_rate(CALLS_EMPTY))

routes_a = ["research_team", "critic", "done"]  # healthy
routes_b = [
    "research_team",
    "research_team",
    "critic",
    "done",
]  # same node twice in a row
routes_c = [
    "research_team",
    "critic",
    "research_team",
    "critic",
]  # ping-pong, never done


def stuck(route: list):
    is_stuck = False
    for i in range(1, len(route)):
        if route[i - 1] == route[i]:
            is_stuck = True
    return is_stuck


# print(stuck(routes_a))
# print(stuck(routes_b))
# print(stuck(routes_c))

import time


def run_code(name):
    time.sleep(0.01)
    return f"[{name}] result"


def traced_run(names, trace: list = []):
    if trace is None:
        trace = []
    for name in names:
        start = time.time()
        result = run_code(name)
        elapsed = time.time() - start
        trace.append({"name": name, "seconds": elapsed})
    return trace


# result_1 = traced_run(["a", "b"])
# print(result_1)
# print(len(result_1))

# result_2 = traced_run(["a", "b"])
# print(result_2)
# print(len(result_2))

EVAL_SET = [
    {
        "task": "pop of france",
        "expect_contains": "68",
        "expect_traj": ["research_team", "done"],
    },
    {
        "task": "report on france",
        "expect_contains": "France",
        "expect_traj": ["research_team", "critic", "done"],
    },
]


def fake_agent(task):  # stands in for your real graph
    canned = {
        "pop of france": (
            "France has about 68 million people.",
            ["research_team", "done"],
        ),
        "report on france": (
            "Report: France ...",
            ["research_team", "research_team", "critic", "done"],
        ),
    }
    return canned[task]  # returns (answer, trajectory)


# for e in EVAL_SET:
#     answer, traj = fake_agent(e["task"])
#     answer_ok = e["expect_contains"] in answer
#     traj_ok = trajectory_ok(
#         traj,
#         e["expect_traj"],
#     )
#     print(e["task"], answer_ok, traj_ok)

from agentic_track.llm import chat  # your real client

case_faithful = {
    "context": "The Eiffel Tower is 330 metres tall.",
    "answer": "The Eiffel Tower is 330 metres tall.",
}
case_grounded_but_false = {
    "context": "The Eiffel Tower is 1000 metres tall.",  # the context is WRONG
    "answer": "The Eiffel Tower is 1000 metres tall.",  # answer faithfully repeats
}
JUDGE_PROMPT = """
You are a groundedness judge.
Decide ONLY whether the ANSWER is supported by the CONTEXT.
Reply with exactly one word: GROUNDED or UNSUPPORTED.
"""


def judge(case):
    user_text = f"CONTEXT: {case['context']} \n ANSWER: {case['answer']}"
    message = [
        {"role": "system", "content": JUDGE_PROMPT},
        {"role": "user", "content": user_text},
    ]
    reply = chat(message)
    return reply.content.strip()


print(judge(case_faithful))
print(judge(case_grounded_but_false))
