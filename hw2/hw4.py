from openai import OpenAI
import tiktoken
from transformers import AutoTokenizer
from dotenv import load_dotenv
import os
import sys

print(sys.path)
load_dotenv(".env")


client = OpenAI(
    base_url="http://localhost:11434/v1/",
    api_key="ollama",
)

# set up a temperature parameter
temperature = 0
prompt = "What's the formula for energy?"
model = "gemma:2b"

response = client.chat.completions.create(
    model=model,
    messages=[{"role": "user", "content": prompt}],
    temperature=temperature,
)

print(f"Completion tokens count is : {response.usage.completion_tokens}")


# # Answer
# print(response.choices[0].message.content)
# access_token = os.getenv('HUG_FACE_TOKEN')
# tokenizer = AutoTokenizer.from_pretrained("google/gemma-2b", token = access_token)

# # Calculate how many tokens the prompt will use
# prompt_tokens = tokenizer.encode(prompt)

# # Calculate how many tokens the response will use
# response_tokens = tokenizer.encode(response.choices[0].message.content)
