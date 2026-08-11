import re, os
from groq import Groq
from dotenv import load_dotenv

load_dotenv()
# 01. the tool registry
# def shout(text):
#     return text.upper()

# def reverse_text(text):
#     return text[::-1]


# TOOLS = {
# "shout": shout,
# "reverse_text": reverse_text,
# }

# fn_1 = TOOLS["shout"]
# result_1 = fn_1("hello world")

# fn_2 = TOOLS["reverse_text"]
# result_2 = fn_2("hello world")

# print(result_1)
# print(result_2)

# 02. dispatch by variable
# def add_one(x):
#     return str(int(x) + 1)

# def double(x):
#     return str(int(x) * 2)

# TOOLS = {
#     "add_one": add_one, 
#     "double": double
#     }

# chosen = "double" # pretend the model picked this
# chosen_2 = "add_one" # pretend the model picked this
# arg = "21" # and passed this

# fn_1 = TOOLS[chosen]
# result = fn_1(arg)
# print("result 1", result)

# fn_2 = TOOLS[chosen_2]
# result_2 = fn_2(arg)
# print("result 2", result_2)

# 03. parse the action



# reply = "Thought: I should multiply.\nAction: calculator[12 * 8]"

# def parse_action(text):
#     match = re.search(r"Action:\s*(\w+)\[(.*)\]",text)
#     if not match:
#         return None, None
#     return match.group(1), match.group(2)
# print(parse_action(reply))

# 04. delete the stop
# reply_a = "Thought: that's it.\nFinal Answer: 12 times 8 is 96."
# reply_b = "Thought: I need a tool.\nAction: calculator[12 * 8]"

# def is_final(text):
#     if "Final Answer:" in text:
#         return text.split("Final Answer:")[-1].strip()
#     else:
#         return "Not final"
# print(is_final(reply_a))
# print(is_final(reply_b))

# 05. guard a bad tool name
# TOOLS = {"calculator": lambda e: str(eval(e))}
# # tool_name = "cal" # the model misspelled it
# tool_name = "calculator"
# tool_input = "2 + 2"

# def guard(text:str):
#     if text in TOOLS:
#         tool = TOOLS[text]
#         return tool(tool_input)
#     else:
#         return f"Error: unknown tool '{text}'"
# print(guard(tool_name))

# 06. one full turn, by hand
# def word_count(text):
#     return str(len(text.split()))

# TOOLS = {"word_count": word_count}

# reply = "Thought: I'll count them.\nAction: word_count[the quick brown fox jumps]"

# def parse_action(text:str):
#     match = re.search(r"Action:\s*(\w+)\[(.*)\]",text)
#     if not match:
#         return None, None
#     return match.group(1), match.group(2)

# action, text = parse_action(reply)
# command = TOOLS[action]

# answer = command(text)
# print("Observation:", answer)


# 07. react system prompt



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

def chat(messages):
    resp = client.chat.completions.create(
        model= "llama-3.3-70b-versatile",
        messages=messages,
        temperature=0
    )
    return resp.choices[0].message.content

def parse_action(text:str):
    match = re.search(r"Action:\s*(\w+)\[(.*)\]",text)
    if not match:
        return None, None
    return match.group(1), match.group(2)

def calculator(expression):
    return str(eval(expression))

def word_count(text):
    return str(len(text.split()))

TOOLS = {"calculator": calculator, "word_count": word_count}

messages = [{"role":"system", "content": SYSTEM_PROMPT},
            ]

def run_agent(question, max_steps = 5):
    messages.append({"role":"user", "content":question})
    for _ in range(max_steps):
        resp = chat(messages)
        messages.append({"role":"assistant", "content": resp})
        if "Final Answer:" in resp:
            return resp.split("Final Answer:")[-1].strip()

        tool_name, tool_ip = parse_action(resp)
        if tool_name in TOOLS:
            answer = TOOLS[tool_name](tool_ip)
        else:
            answer = f"Error: unknown tool name '{tool_name}'"

        messages.append({"role":"user", "content":f"Observation: {answer}"})
    return "Stopped: hit max rate without final answer."

print(run_agent("What is 47 * 89, and how many words are in 'the agent loop is beating'?"))
        
