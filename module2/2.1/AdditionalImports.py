import os
from dotenv import load_dotenv
from smolagents import CodeAgent,tool,InferenceClientModel
from smolagents import CodeAgent, InferenceClientModel
import numpy as np
import time
import datetime


# Load environment variables from .env file
load_dotenv(os.path.join(os.path.dirname(__file__), '../../.env'))

# Get HuggingFace token from environment
hf_token = os.getenv('HF_TOKEN')



agent = CodeAgent(tools=[], model=InferenceClientModel(model_id="Qwen/Qwen2.5-Coder-32B-Instruct", token=hf_token), additional_authorized_imports=['datetime'])

agent.run(
    """
    Zain needs to prepare for the party. Here are the tasks:
    1. Prepare the drinks - 30 minutes
    2. Decorate the house - 60 minutes
    3. Set up the menu - 45 minutes
    4. Prepare the music and playlist - 45 minutes

    If we start right now, at what time will the party be ready?

    """
)
