"""Parse the model's text reply into a (tool_name, tool_input) pair. Brittle by design — MCP replaces this in Module 2."""

import re


def parse_action(text):
    match = re.search(r"Action:\s*(\w+)\[(.*)\]", text)
    if not match:
        return None, None
    return match.group(1), match.group(2)