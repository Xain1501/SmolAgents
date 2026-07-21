from smolagents import ToolCallingAgent, WebSearchTool, InferenceClientModel
import os
from dotenv import load_dotenv


# Load environment variables from .env file
load_dotenv(os.path.join(os.path.dirname(__file__), '../../.env'))

# Get HuggingFace token from environment
hf_token = os.getenv('HF_TOKEN')



agent = ToolCallingAgent(tools=[WebSearchTool()], model=InferenceClientModel(model_id="Qwen/Qwen2.5-72B-Instruct", token=hf_token))

agent.run("Search for the best music recommendations for a party at wayne mansion")