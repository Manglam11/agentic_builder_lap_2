"""Multi-agent orchestration as nested LangGraphs (M5 weld).

Two graphs, one shape. The CHILD (`build_research_team`) is the P7
graph: a supervisor routes between researcher and writer until every
asked-for step has a matching [agent] label, then END. The PARENT
(`build_supervisor_graph`) is the P8 graph-of-graphs: it treats the
whole child team as a single node, then adds a critic.

Design notes carried from the puzzle arc:
- Node contract: input = full `state` dict, output = a state-update
  dict, never a bare route. The routing decision is STASHED in
  `state["next"]` by a supervisor node, then READ by the pure `route`
  function the conditional edge calls.
- The conditional-edge map holds node-NAME strings, not function objects.
- Reducer is `operator.add` (plain-dict concat), matching graph.py —
  Groq-native throughout, no add_messages translation tax.
- Node bodies are built by factories (make_specialist / make_supervisor):
  five near-identical hand-written nodes collapse into two patterns
  defined once. A closure bakes in each node's prompt + label.
- Encapsulation: research_team_node copies up ONLY the child's final
  result, re-labeled at the parent's vocabulary ([research_team]);
  child scratch-work ([researcher]/[writer]) stays sealed inside.

KNOWN WEAKNESS (design debt): the parent's label-checklist gets buried
under report text, so it can pick research_team twice. When the real
M3/M4 specialists go in, track completed steps in a dedicated state
field instead of parsing tags out of prose (tag-parsing doesn't scale).
"""

import operator
import re
from typing import Annotated, TypedDict

from langgraph.graph import END, START, StateGraph

from agentic_track.llm import chat
from agentic_track.prompts import (
    CRITIC_PROMPT,
    PARENT_SUPERVISOR_PROMPT,
    RESEARCHER_PROMPT,
    SUPERVISOR_PROMPT,
    WRITER_PROMPT,
)


class SupervisorState(TypedDict):
    messages: Annotated[list, operator.add]
    next: str


def strip_label(text: str) -> str:
    """Drop a leading [agent] tag so copied-up / final text reads clean.
    Regex handles ANY tag — no need to enumerate every specialist name."""
    return re.sub(r"^\[[^\]]+\]\s*", "", text)


def route(state: SupervisorState) -> str:
    """Pure path function: read the stashed decision, return the edge key.
    Shared by both graphs — both stash their choice in state['next']."""
    return state["next"]


# ── node factories: two patterns, built once, reused everywhere ──
def make_specialist(prompt: str, label: str):
    """Build a worker node: run one prompt, label the reply with [label]."""

    def node(state: SupervisorState) -> dict:
        messages = [{"role": "system", "content": prompt}] + state["messages"]
        reply = chat(messages)
        return {
            "messages": [{"role": "assistant", "content": f"[{label}] {reply.content}"}]
        }

    return node


def make_supervisor(prompt: str):
    """Build a router node: stash a normalized one-word decision in state['next'].
    .strip().lower() guards against 'Researcher.' / 'DONE' missing the edge map."""

    def node(state: SupervisorState) -> dict:
        messages = [{"role": "system", "content": prompt}] + state["messages"]
        return {"next": chat(messages).content.strip().lower()}

    return node


# ── child team: researcher + writer under a supervisor (P7) ──
def build_research_team():
    g = StateGraph(SupervisorState)
    g.add_node("supervise", make_supervisor(SUPERVISOR_PROMPT))
    g.add_node("researcher", make_specialist(RESEARCHER_PROMPT, "researcher"))
    g.add_node("writer", make_specialist(WRITER_PROMPT, "writer"))
    g.add_edge(START, "supervise")
    g.add_conditional_edges(
        "supervise",
        route,
        {"researcher": "researcher", "writer": "writer", "done": END},
    )
    g.add_edge("researcher", "supervise")
    g.add_edge("writer", "supervise")
    return g.compile()


# ── parent team: research_team (as one node) + critic (P8) ──
def build_supervisor_graph():
    """Public door. Build the child, wrap it as a node, add a critic,
    return the compiled parent app. Mirrors graph.build_graph."""
    research_team = build_research_team()

    def research_team_node(state: SupervisorState) -> dict:
        sub = research_team.invoke(
            {"messages": state["messages"]},
            {"recursion_limit": 6},
        )
        clean = strip_label(sub["messages"][-1]["content"])
        return {
            "messages": [{"role": "assistant", "content": f"[research_team] {clean}"}]
        }

    parent = StateGraph(SupervisorState)
    parent.add_node("parent_supervise", make_supervisor(PARENT_SUPERVISOR_PROMPT))
    parent.add_node("research_team", research_team_node)
    parent.add_node("critic", make_specialist(CRITIC_PROMPT, "critic"))
    parent.add_edge(START, "parent_supervise")
    parent.add_conditional_edges(
        "parent_supervise",
        route,
        {"research_team": "research_team", "critic": "critic", "done": END},
    )
    parent.add_edge("research_team", "parent_supervise")
    parent.add_edge("critic", "parent_supervise")
    return parent.compile()


def run_supervisor(question, recursion_limit=8):
    app = build_supervisor_graph()
    result = app.invoke(
        {"messages": [{"role": "user", "content": question}]},
        {"recursion_limit": recursion_limit},
    )
    return strip_label(result["messages"][-1]["content"])


if __name__ == "__main__":
    print(
        run_supervisor(
            "Research the printing press's economic impact, "
            "write a short report, then critique it."
        )
    )
