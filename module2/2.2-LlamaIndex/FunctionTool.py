from llama_index.core.tools import FunctionTool
from llama_index.core.agent.workflow import FunctionAgent
from llama_index.llms.huggingface_api import HuggingFaceInferenceAPI
import os
from dotenv import load_dotenv
import asyncio

load_dotenv()


llm = HuggingFaceInferenceAPI(
    model_name="Qwen/Qwen2.5-7B-Instruct",
    token=os.getenv("HF_TOKEN"),
    provider="hf-inference",
)


def get_weather(location: str) -> str:
    """Useful for getting the weather for a given location."""
    print(f"Getting weather for {location}")
    return f"The weather in {location} is sunny"

tool = FunctionTool.from_defaults(
    get_weather,
    name="my_weather_tool",
    description="Useful for getting the weather for a given location.",
)

tool.call("New York")

agent = FunctionAgent(
    llm=llm,
    tools=[tool],
)

async def main():
    response = await agent.run(
        "What's the weather in New York?"
    )

    print(response)

asyncio.run(main())



# <the reason this code isnt working is that i dont have a function calling llm model change model with openai and openapikey to make this work>