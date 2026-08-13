"""The MCP boundary — connects to the tool server and translates its
catalog into the schema Groq's tool-calling expects.

Mirror of llm.py: that's the boundary to the model, this is the boundary
to the tools. The server is the single source of truth for tool schemas.
"""

from pathlib import Path

from fastmcp import Client

SERVER = Path(__file__).parent / "tools.py"
mcp_client = Client(str(SERVER))


async def load_tools_from_mcp(client):
    """Ask the server what tools it offers, return them as Groq function schemas."""
    mcp_tools = await client.list_tools()
    return [
        {
            "type": "function",
            "function": {
                "name": t.name,
                "description": t.description,
                "parameters": t.inputSchema,  # server already made this a JSON schema
            },
        }
        for t in mcp_tools
    ]
