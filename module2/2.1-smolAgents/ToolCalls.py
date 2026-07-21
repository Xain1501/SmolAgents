import os
from dotenv import load_dotenv
from smolagents import CodeAgent,Tool,InferenceClientModel


# Load environment variables from .env file
load_dotenv(os.path.join(os.path.dirname(__file__), '../../.env'))

# Get HuggingFace token from environment
hf_token = os.getenv('HF_TOKEN')


class SuperheroPartyThemeTool(Tool):
  name="superhero_Party_theme_generator"
  description= """
  This tool suggests creative superhero-themed party ideas based on a category. 
  It returns a unique party theme idea.
 """

  inputs= {
    "category":{
      "type":"string",
      "description":"The type of super hero party (e.g, 'classic heroes' , 'villian masquerade', 'futuristic gotham ').",
  } 
 }

  output_type="string"

  def forward(self, category: str):
        themes = {
            "classic heroes": "Justice League Gala: Guests come dressed as their favorite DC heroes with themed cocktails like 'The Kryptonite Punch'.",
            "villain masquerade": "Gotham Rogues' Ball: A mysterious masquerade where guests dress as classic Batman villains.",
            "futuristic Gotham": "Neo-Gotham Night: A cyberpunk-style party inspired by Batman Beyond, with neon decorations and futuristic gadgets."
        }

        return themes.get(category.lower(), "Themed party idea not found. Try 'classic heroes', 'villain masquerade', or 'futuristic Gotham'.")
  
party_theme_tool = SuperheroPartyThemeTool()
agent = CodeAgent(tools=[party_theme_tool], model=InferenceClientModel(model_id="Qwen/Qwen2.5-Coder-32B-Instruct", token=hf_token))


result=agent.run("What would be a good superhero party idea for a 'villian masquerade' theme?")

print(result)