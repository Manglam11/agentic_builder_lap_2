import json
from types import SimpleNamespace
from groq import Groq
from dotenv import load_dotenv

load_dotenv()

client = Groq()


def get_weather(city):
    return f"{city}: 27C, clear"


tools = [
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Get the current weather for a given city.",
            "parameters": {
                "type": "object",
                "properties": {
                    "city": {"type": "string", "description": "A city name, eg Mumbai"}
                },
                "required": ["city"],
            },
        },
    }
]


TOOLS = {"get_weather": get_weather}

fake_call = SimpleNamespace(
    id="call_1",
    function=SimpleNamespace(name="get_weather", arguments='{"city": "Delhi"}'),
)


# name = fake_call.function.name
# # args = fake_call.function.arguments
# args = json.loads(fake_call.function.arguments)
# result = TOOLS[name](**args)
# # print(result)

messages = [
    {
        "role": "system",
        "content": "Answer using only the tool result. One sentence, no extra commentary.",
    },
    {"role": "user", "content": "weather in Mumbai"},
]

# assistant_msg = {
#     "role": "assistant",
#     "content": None,
#     "tool_calls": [fake_call],
# }

# tool_result = "Delhi: 27C, clear"

# messages.append(assistant_msg)
# messages.append({"role": "tool", "tool_call_id": fake_call.id, "content": str(result)})
# print(messages)

response = client.chat.completions.create(
    model="llama-3.3-70b-versatile",
    messages=messages,
    tools=tools,
    tool_choice="auto",
    temperature=0,
)

msg = response.choices[0].message

if msg.tool_calls:
    messages.append(msg)
    for tc in msg.tool_calls:
        name = tc.function.name
        args = json.loads(tc.function.arguments)
        result = TOOLS[name](**args)
        messages.append({"role": "tool", "tool_call_id": tc.id, "content": str(result)})
else:
    print(msg.content)


response2 = client.chat.completions.create(
    model="llama-3.3-70b-versatile", messages=messages
)

result2 = response2.choices[0].message.content

print(result2)
print("FIRST msg.tool_calls:", msg.tool_calls)
print("FIRST msg.content:", msg.content)
