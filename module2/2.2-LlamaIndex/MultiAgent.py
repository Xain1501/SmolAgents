
from llama_index.core.agent.workflow import (
    AgentWorkflow,
    ToolCallResult,
    AgentStream,
    ReActAgent,
)
import asyncio
import os
from dotenv import load_dotenv

import chromadb

from llama_index.core import VectorStoreIndex
from llama_index.llms.huggingface_api import HuggingFaceInferenceAPI
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core.tools import QueryEngineTool
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.core.tools import FunctionTool



load_dotenv()
load_dotenv(os.path.join(os.path.dirname(__file__), "../../.env"))

hf_token = os.getenv("HF_TOKEN")

llm = HuggingFaceInferenceAPI(
    model_name="Qwen/Qwen2.5-Coder-32B-Instruct",
    temperature=0.7,
    max_tokens=100,
    token=hf_token,
    provider="auto",
)




# Create a vector store
db = chromadb.PersistentClient(path="./alfred_chroma_db")
chroma_collection = db.get_or_create_collection("alfred")
vector_store = ChromaVectorStore(chroma_collection=chroma_collection)

# Create a query engine
embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-small-en-v1.5")

index = VectorStoreIndex.from_vector_store(
    vector_store=vector_store, embed_model=embed_model
)

query_engine = index.as_query_engine(llm=llm)

query_engine_tool = QueryEngineTool.from_defaults(
    query_engine=query_engine,
    name="personas",
    description="descriptions for various types of personas",
    return_direct=False,
)




def add(a: int, b: int) -> int:
    """Add two numbers."""
    return a + b


def subtract(a: int, b: int) -> int:
    """Subtract two numbers."""
    return a - b


def multiply(a: int, b: int) -> int:
    """Multiply two numbers."""
    return a * b


def divide(a: int, b: int) -> float:
    """Divide two numbers."""
    return a / b


calculator_agent = ReActAgent(
    name="calculator",
    description=" An Agent that can perform addition subtraction multiplication and division",
    tools=[
        FunctionTool.from_defaults(add),
        FunctionTool.from_defaults(subtract),
        FunctionTool.from_defaults(multiply),
        FunctionTool.from_defaults(divide),
    ],
    llm=llm,
    system_prompt=(
        "You are a math agent. "
        "Always use the provided math tools to solve calculations."
    ),
)

query_agent = ReActAgent(
    name="info_lookup",
    description="Looks up information about heroes ",
    system_prompt="You are a helpful assistant that has access to a database containing persona descriptions.",
    tools=[query_engine_tool],
    llm=llm,
)






agent = AgentWorkflow(agents=[calculator_agent, query_agent], root_agent="calculator")

async def main():
    handler = agent.run(user_msg="Can you perform (2 + 2) * 2 ?")

    async for ev in handler.stream_events():
        if isinstance(ev, ToolCallResult):
            print()
            print("Tool Called:", ev.tool_name)
            print("Arguments :", ev.tool_kwargs)
            print("Output    :", ev.tool_output)

        elif isinstance(ev, AgentStream):
            print(ev.delta, end="", flush=True)

    response = await handler

    print("\nFinal Response:")
    print(response)


if __name__ == "__main__":
    asyncio.run(main())


    # documented painpoint in qwen models causing issues with streamlining and multiagent code for this code to work use openapi key and model