from openai import OpenAI
from dotenv import load_dotenv
import os 
import tiktoken

# model = 'gpt-4o' # gpt-3.5-turbo

def llm(prompt, model='gpt-4o'):

    # bring in the openai key 
    load_dotenv('.env')
    key = os.environ.get("OPENAI_API_KEY")
    client = OpenAI(api_key=key)
    
    # Calculate how many tokens the prompt will use
    encoding = tiktoken.encoding_for_model(model)
    prompt_tokens = encoding.encode(prompt)
    
    print(f"Prompt tokens: {len(prompt_tokens)}")

    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}]
    )
    
    # Calculate how many tokens the response will use
    response_tokens = encoding.encode(response.choices[0].message.content)
    print(f"Response tokens: {len(response_tokens)}")
    
    return response.choices[0].message.content