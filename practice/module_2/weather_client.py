import asyncio

from fastmcp import Client

client = Client("weather_server.py")


async def main():
    async with client:
        tools = await client.list_tools()
        print("Tool available:", [t.name for t in tools])
        result1 = await client.call_tool("get_weather", {"city": "delhi"})

        result2 = await client.call_tool("word_count", {"text": "Namashkaram"})
        print(result1.data)
        print(result2.data)


asyncio.run(main())
