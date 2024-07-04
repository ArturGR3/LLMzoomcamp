from prompt_building import build_prompt
from min_search import minisearch
from modify_doc import modify_docs
from elastic_search import elastic_search
from open_ai_prompt import llm
from gemini_api import llm_gemini
import click
import json

# course = 'machine-learning-zoomcamp'
# Bring in document name and query from command line 
@click.command()
@click.option('--document', prompt="Provide a document to parse ",  help='The document file')
@click.option('--query', prompt = "How can I help you", help='The query to chat bot')
@click.option('--course', prompt = "Course?", help='Based on what course you want to search the document?')

def rag(document, query, course):
    """
    OpenAI API response to a query based on the search results from MiniSearch and Elasticsearch
    """
    # Load the documents from the JSON file
    modified_document = modify_docs(doc = document)
    
    # Create and search results from Minisearch 
    mini_search_result = minisearch(query, document = modified_document, course=course, num_results = 3)
    
    # Create and search results from Elasticsearchdas
    elastic_search_result = elastic_search(query, document = modified_document, course = course, num_results = 3)
    
    # Create a prompt for the OpenAI API based on the search results of MiniSearch and Elasticsearch
    min_search_prompt       = build_prompt(query, mini_search_result)
    print (f"the length of the prompt based on Min Search is : {len(min_search_prompt)}")
    
    elastic_search_prompt   = build_prompt(query, elastic_search_result)
    print (f"the length of the prompt based on Elastic Search is :  {len(elastic_search_prompt)}")
        
    # Return the response from the OpenAI API
    print("Min search based prompt")
    min_response     = llm_gemini(min_search_prompt)
    # min_response     = llm(min_search_prompt) # This is for OpenAI API
    print(f"Min search response via OpenAI API:\n{min_response}")
    
    print('-----------------')
    print('-----------------')
    
    print("Elastic search based prompt")
    elastic_response = llm_gemini(elastic_search_prompt)
    # elastic_response = llm(elastic_search_prompt) # This is for OpenAI API
    print(f"Elastic search response via OpenAI API:\n{elastic_response}")
    # Run this function from the command line
if __name__ == '__main__':
    rag() 
