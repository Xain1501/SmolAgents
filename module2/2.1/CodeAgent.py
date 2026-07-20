import os
from dotenv import load_dotenv
from smolagents import CodeAgent, WebSearchTool, InferenceClientModel

# Load environment variables from .env file
load_dotenv(os.path.join(os.path.dirname(__file__), '../../.env'))

# Get HuggingFace token from environment
hf_token = os.getenv('HF_TOKEN')

agent = CodeAgent(tools=[WebSearchTool()], model=InferenceClientModel(model_id="Qwen/Qwen2.5-Coder-32B-Instruct", token=hf_token))

agent.run("Search for the best food recommendations for a party at thee Governors mansion")
