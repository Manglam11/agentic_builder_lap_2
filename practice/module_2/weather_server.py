from fastmcp import FastMCP

mcp = FastMCP("weather-tools")


@mcp.tool
def get_weather(city: str) -> str:
    """Get the current weather for a given city."""
    return f"{city} has a temperature of 27 degrees Celsius with clear weather."


@mcp.tool
def word_count(text: str) -> str:
    """Count the number of words in a piece of text."""
    text_lenght = len(text.split())
    return f"Text '{text}' is '{text_lenght}' words long."


if __name__ == "__main__":
    mcp.run()
