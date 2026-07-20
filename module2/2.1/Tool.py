import os
from dotenv import load_dotenv
from smolagents import CodeAgent,tool,InferenceClientModel

# Load environment variables from .env file
load_dotenv(os.path.join(os.path.dirname(__file__), '../../.env'))

# Get HuggingFace token from environment
hf_token = os.getenv('HF_TOKEN')

@tool
def suggestGames(Weather:str)-> str:
  """
  Suggests the type of Games that could be played  based on the weather
  Args:
    Weather(str): The Weather Prediction for the date on which the Games are planned. Allowed values are :
    - "Sunny": Something outdoors example swimming
    - "Windy": Some kind of sport that uses wind example kite flying
    - "rainy": Some kind of indoor games example board games
    - "Custom": Voting amongst the players on what to play 

  """
  if Weather == "Sunny":
    return "swimming waterpolo volleyball"
  elif Weather == "Windy":
    return "flying kites , racing paper airplanes , frisbee"
  elif Weather == "rainy":
    return " bluff, carrom, ludo "
  else:
    return "Decide on a list of games and then vote to choose what to play"

agent = CodeAgent(tools=[suggestGames], model=InferenceClientModel(model_id="Qwen/Qwen2.5-Coder-32B-Instruct", token=hf_token))

agent.run("What games can 5 people play on a windy day")
