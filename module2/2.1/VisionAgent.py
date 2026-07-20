from PIL import Image
import requests
from io import BytesIO

from smolagents import CodeAgent, InferenceClientModel

import os
from dotenv import load_dotenv

load_dotenv(os.path.join(os.path.dirname(__file__), '../../.env'))


hf_token = os.getenv('HF_TOKEN')

model = InferenceClientModel(
    model_id="Qwen/Qwen2.5-VL-72B-Instruct",
    token=hf_token,
)
image_urls = [
    "https://upload.wikimedia.org/wikipedia/commons/e/e8/The_Joker_at_Wax_Museum_Plus.jpg", # Joker image
    "https://upload.wikimedia.org/wikipedia/en/9/98/Joker_%28DC_Comics_character%29.jpg" # Joker image
]



images = []
for url in image_urls:
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/114.0.0.0 Safari/537.36" 
    }
    response = requests.get(url,headers=headers)
    image = Image.open(BytesIO(response.content)).convert("RGB")
    images.append(image)



# Instantiate the agent
agent = CodeAgent(
    tools=[],
    model=model,
)

response = agent.run(
    """
    Describe the costume and makeup that the comic character in these photos is wearing and return the description.
    Tell me if the guest is The Joker or Wonder Woman.
    """,
    images=images
)

