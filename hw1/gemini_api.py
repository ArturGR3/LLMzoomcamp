import pathlib
import textwrap
import google.generativeai as genai
from IPython.display import display
from IPython.display import Markdown
from dotenv import load_dotenv
import os


def to_markdown(text):
    text = text.replace("•", "  *")
    return Markdown(textwrap.indent(text, "> ", predicate=lambda _: True))


model_gemini = "gemini-1.5-pro"


def llm_gemini(prompt, model=model_gemini):

    # bring in the openai key
    load_dotenv(".env")
    key = os.environ.get("GOOGLE_API_KEY")
    client = genai.configure(api_key=key)
    model = genai.GenerativeModel(model)

    response = model.generate_content(prompt)
    return response.text
