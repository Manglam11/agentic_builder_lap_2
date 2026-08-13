"""The agent's tools, exposed as an MCP server.

Each @mcp.tool function's type hints become its input schema and its
docstring becomes its description — FastMCP generates both for the model.
"""

from fastmcp import FastMCP

mcp = FastMCP("agentic-track-tools")


@mcp.tool
def calculator(expression: str) -> str:
    """Evaluate a math expression and return the result."""
    return str(eval(expression))  # eval wart — tracked for M7 guardrails


@mcp.tool
def word_count(text: str) -> str:
    """Count the number of words in a piece of text."""
    return str(len(text.split()))


if __name__ == "__main__":
    mcp.run()
