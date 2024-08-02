import requests 
from tqdm.auto import tqdm
import pandas as pd
from sentence_transformers import SentenceTransformer
from elasticsearch import Elasticsearch
from openai import OpenAI
import json 
import os 
from dotenv import load_dotenv, find_dotenv

load_dotenv(find_dotenv(usecwd=True))
 
base_url = 'https://github.com/DataTalksClub/llm-zoomcamp/blob/main'
relative_url = '03-vector-search/eval/documents-with-ids.json'
docs_url = f'{base_url}/{relative_url}?raw=1'
docs_response = requests.get(docs_url)
documents = docs_response.json()

base_url = 'https://github.com/DataTalksClub/llm-zoomcamp/blob/main'
relative_url = '03-vector-search/eval/ground-truth-data.csv'
ground_truth_url = f'{base_url}/{relative_url}?raw=1' 

df_ground_truth = pd.read_csv(ground_truth_url) 
df_ground_truth = df_ground_truth[df_ground_truth.course == 'machine-learning-zoomcamp']
ground_truth = df_ground_truth.to_dict(orient='records') 

doc_idx = {d['id']: d for d in documents} # Index documents by id
doc_idx['5170565b']['text']


model_name = 'multi-qa-MiniLM-L6-cos-v1' 
model = SentenceTransformer(model_name)

# Elastic search has to run on the docker container
es_client = Elasticsearch('http://localhost:9200') 

# Create index
index_settings = {
    "settings": {
        "number_of_shards": 1,
        "number_of_replicas": 0
    },
    "mappings": {
        "properties": {
            "text": {"type": "text"},
            "section": {"type": "text"},
            "question": {"type": "text"},
            "course": {"type": "keyword"},
            "id": {"type": "keyword"},
            "question_text_vector": {
                "type": "dense_vector",
                "dims": 384,
                "index": True,
                "similarity": "cosine"
            },
        }
    }
}

index_name = "course-questions"

# Delete index if exists
es_client.indices.delete(index=index_name, ignore_unavailable=True)
# Create index based on settings
es_client.indices.create(index=index_name, body=index_settings)


# Index documents
for doc in tqdm(documents):
    question = doc['question']
    text = doc['text']
    # Encode question and text to get the vector representation for indexing
    doc['question_text_vector'] = model.encode(question + ' ' + text)
    # Create index
    es_client.index(index=index_name, document=doc)
    
# Search for similar questions based on the question vector    
def elastic_search_knn(field, vector, course): 
    """
    Search for similar questions based on the question vector
    :param field: field name
    :param vector: question vector
    :param course: course name
    
    :return: list of similar questions    
    """
    # Define knn query
    knn = {
        "field": field,
        "query_vector": vector, 
        "k": 5, # Number of similar questions to retrieve
        "num_candidates": 10000, # Number of candidates to retrieve
        "filter": {
            "term": {
                "course": course # Filter by course
            }
        }
    }

    search_query = {
        "knn": knn, # Define knn query
        "_source": ["text", "section", "question", "course", "id"] # Return only these fields
    }

    es_results = es_client.search(
        index=index_name,
        body=search_query
    )
    
    result_docs = [] # List of similar questions
    
    for hit in es_results['hits']['hits']: # Iterate over search results
        result_docs.append(hit['_source']) # 

    return result_docs

def question_text_vector_knn(q):
    question = q['question']
    course = q['course']

    v_q = model.encode(question)

    return elastic_search_knn('question_text_vector', v_q, course)


question_text_vector_knn(dict(
    question='Are sessions recorded if I miss one?',
    course='machine-learning-zoomcamp'
))


def build_prompt(query, search_results):
    prompt_template = """
You're a course teaching assistant. Answer the QUESTION based on the CONTEXT from the FAQ database.
Use only the facts from the CONTEXT when answering the QUESTION.

QUESTION: {question}

CONTEXT: 
{context}
""".strip()

    context = ""
    
    for doc in search_results:
        context = context + f"section: {doc['section']}\nquestion: {doc['question']}\nanswer: {doc['text']}\n\n"
    
    prompt = prompt_template.format(question=query, context=context).strip()
    return prompt


client = OpenAI()

def llm(prompt, model='gpt-4o-mini'):
    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}]
    )
    
    return response.choices[0].message.content


def rag(query: dict, model='gpt-4o') -> str:
    """
    Retrieve, Answer, Generate (RAG) model
    
    Args:
    query: dict - query with question and course
    model: str - model name from OpenAI
    """
    # Search for similar questions based on the question vector using Elastic Search
    search_results = question_text_vector_knn(query)
    print(json.dumps(search_results, indent=2))
    # Build prompt for the model based on the search results
    prompt = build_prompt(query['question'], search_results)
    # print(prompt)
    # Generate answer using the RAG model
    answer = llm(prompt, model=model)
    return answer

ground_truth[10]
rag(ground_truth[10])
doc_idx['5170565b']


answer_orig = 'Yes, sessions are recorded if you miss one. Everything is recorded, allowing you to catch up on any missed content. Additionally, you can ask questions in advance for office hours and have them addressed during the live stream. You can also ask questions in Slack.'
answer_llm = 'Everything is recorded, so you won’t miss anything. You will be able to ask your questions for office hours in advance and we will cover them during the live stream. Also, you can always ask questions in Slack.'


v_llm = model.encode(answer_llm)
v_orig = model.encode(answer_orig)

# Measure similarity between the original and generated answers
# As the vectors are normalized, we can use dot product as a similarity measure
v_llm.dot(v_orig) 

# answers = {}

# for i, rec in tqdm(enumerate(ground_truth)):
#     if i in answers:
#         continue
    
#     rec = ground_truth[1]
#     document_id = rec['document']
#     doc = doc_idx[document_id]
    
    
#     answer_llm = rag(rec)
#     answer_orig = doc['text']
    
#     answers[i] = {
#         'answer_llm': answer_llm,
#         'answer_orig': answer_orig,
#         'document': document_id,
#         'question': rec['question'],
#         'course': rec['course'],  
#     }
    

results_gpt4o = pd.read_csv('results_gpt4o.csv')


def compute_similarity(record):
    answer_orig = record['answer_orig']
    answer_llm = record['answer_llm']
    
    v_orig = model.encode(answer_orig)
    v_llm = model.encode(answer_llm)
        
    return v_orig.dot(v_llm)

similarity = []
# Measure similarity between the original and generated answers for first 10 records
for i in range(10):
    record = results_gpt4o.iloc[i]
    similarity.append(compute_similarity(record))