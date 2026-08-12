"""The agent loop — wires prompt + llm + tools + parsing into one reusable ReAct agent."""

from agentic_track.prompts import SYSTEM_PROMPT
from agentic_track.llm import chat
from agentic_track.tools import TOOLS
from agentic_track.parsing import parse_action


def run_agent(question, max_steps=5):
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": question},
    ]
    for step in range(max_steps):
        resp = chat(messages)
        messages.append({"role": "assistant", "content": resp})

        if "Final Answer:" in resp:
            return resp.split("Final Answer:")[-1].strip()

        tool_name, tool_input = parse_action(resp)
        if tool_name in TOOLS:
            answer = TOOLS[tool_name](tool_input)
        else:
            answer = f"Error: unknown tool name '{tool_name}'"

        messages.append({"role": "user", "content": f"Observation: {answer}"})

    return "Stopped: hit max steps without final answer."


if __name__ == "__main__":
    print(run_agent("What is 47 * 89, and how many words are in 'the agent loop is beating'?"))