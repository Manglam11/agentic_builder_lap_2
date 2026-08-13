"""Agent state — one bag threaded through the loop, plus the windowing helper."""

from typing import TypedDict

from agentic_track.prompts import SYSTEM_PROMPT


class AgentState(TypedDict):
    messages: list  # short-term memory: the running transcript
    memory: list  # long-term hits recalled for this turn
    step: int
    done: bool


def new_state(question):
    return {
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": question},
        ],
        "memory": [],
        "step": 0,
        "done": False,
    }


def trim_messages(messages, max_turns=20):
    """Keep the system message pinned + the most recent `max_turns` turns.

    Reach for this in long multi-turn sessions where history would overflow the
    context window. NOT used by the bounded agent loop (it never overflows at
    max_steps=5). Caveat: on tool-calling histories, don't cut between an
    assistant message with tool_calls and its tool replies — the API pairs them.
    """
    system = messages[:1]
    rest = messages[1:]
    return system + rest[-max_turns:]
