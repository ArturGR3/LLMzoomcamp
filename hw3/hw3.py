from sentence_transformers import SentenceTransformer
import requests
import numpy as np
import json
from tqdm import tqdm
import pandas as pd
from elasticsearch import Elasticsearch

model_name = "multi-qa-distilbert-cos-v1"
embedding_model = SentenceTransformer(model_name)
user_question = "I just discovered the course. Can I still join it?"
v_user_q = embedding_model.encode(user_question)

# Q1
print(f"First value of encoded user question: {v_user_q[0]}")

base_url = "https://github.com/DataTalksClub/llm-zoomcamp/blob/main"
relative_url = "03-vector-search/eval/documents-with-ids.json"
docs_url = f"{base_url}/{relative_url}?raw=1"
docs_response = requests.get(docs_url)
documents = docs_response.json()

# Indent documents(json) for better readability
documents_str = json.dumps(documents, indent=2)
# Filter out the documents and based on the couse = "machine-learning-zoomcamp"
documents_ml = [
    doc for doc in documents if doc["course"] == "machine-learning-zoomcamp"
]
print(f"Number of documents for 'machine-learning-zoomcamp': {len(documents_ml)}")

# Q2. Create a matrix of embedings and add the embeddings to the documents
X = []
modified_documents = []
for doc in tqdm(documents_ml):
    qa_text = f"{doc['question']} {doc['text']}"
    v_qa = embedding_model.encode(qa_text)
    doc["text_vector"] = v_qa.tolist()
    modified_documents.append(doc)
    X.append(v_qa)

X = np.array(X)
print(f"Shape of embeddings array: {X.shape}")

for doc in tqdm(documents_ml):
    doc["text_vector"] = X[doc["id"]]

# Q3
# The highest score between the user question and the documents
highest_score = X.dot(v_user_q).max()
print(f"Highest score between user question and documents: {highest_score:.2f}")


## Vector search
class VectorSearchEngine:
    def __init__(self, documents, embeddings):
        self.documents = documents
        self.embeddings = embeddings

    def search(self, v_query, num_results=10):
        scores = self.embeddings.dot(v_query)
        idx = np.argsort(-scores)[:num_results]
        return [self.documents[i] for i in idx]


base_url = "https://github.com/DataTalksClub/llm-zoomcamp/blob/main"
relative_url = "03-vector-search/eval/ground-truth-data.csv"
ground_truth_url = f"{base_url}/{relative_url}?raw=1"

df_ground_truth = pd.read_csv(ground_truth_url)
df_ground_truth = df_ground_truth[df_ground_truth.course == "machine-learning-zoomcamp"]
ground_truth = df_ground_truth.to_dict(orient="records")

search_engine = VectorSearchEngine(documents=documents_ml, embeddings=X)


# Create a function to calculate the hit rate
def hit_rate(search_engine, user_question, num_results=5):
    question = user_question["question"]
    doc_id = user_question["document"]

    # print(f"Processing question: {question}, doc_id: {doc_id}")
    v_user_q = embedding_model.encode(question)
    results = search_engine.search(v_user_q, num_results=num_results)

    # print(f"Search true document:{doc_id} in {[result['id'] for result in results]}")
    hit = any(result["id"] == doc_id for result in results)
    # print(f"Hit: {hit}")
    return hit


# Calculate hit rate for VectorSearchEngine on all the questions documents_ml
hit_rates = [hit_rate(search_engine, doc) for doc in tqdm(ground_truth)]
final_hit_rate = np.mean(hit_rates)
print(f"Final hit rate: {final_hit_rate}")

# Q5. Indexing with Elasticsearch


es_client = Elasticsearch("http://localhost:9200")

index_settings = {
    "settings": {"number_of_shards": 1, "number_of_replicas": 0},
    "mappings": {
        "properties": {
            "text": {"type": "text"},
            "section": {"type": "text"},
            "question": {"type": "text"},
            "course": {"type": "keyword"},
            "id": {"type": "keyword"},
            "text_vector": {
                "type": "dense_vector",
                "dims": 768,
                "index": True,
                "similarity": "cosine",
            },
        }
    },
}

index_name = "course-questions"

es_client.indices.delete(index=index_name, ignore_unavailable=True)
es_client.indices.create(index=index_name, body=index_settings)


# Create the index with the same settings as in the module (but change the dimensions)
# Index the embeddings (note: you've already computed them)
for document in modified_documents:
    es_client.index(index=index_name, document=document)


def perform_vector_search(
    es_client, index_name, query, search_type="cosine", k=10, num_candidates=1000
):
    v_user_q = embedding_model.encode(query)

    if search_type == "cosine":
        query_elastic_search = {
            "query": {
                "script_score": {
                    "query": {"match_all": {}},
                    "script": {
                        "source": "cosineSimilarity(params.query_vector, 'text_vector') + 1.0",
                        "params": {"query_vector": v_user_q},
                    },
                }
            },
            "size": k,  # Set the number of results to return for cosine search
        }
    elif search_type == "knn":
        query_elastic_search = {
            "knn": {
                "field": "text_vector",
                "query_vector": v_user_q,
                "k": k,
                "num_candidates": num_candidates,
            }
        }
    else:
        raise ValueError("Invalid search_type. Must be 'cosine' or 'knn'.")

    es_result = es_client.search(index=index_name, body=query_elastic_search)
    result_docs = []

    for hit in es_result["hits"]["hits"]:
        result_docs.append(hit["_source"])
        # add the score to the result
        result_docs[-1]["score"] = hit["_score"]

    return result_docs


# Example usage:
user_question = "I just discovered the course. Can I still join it?"
results = perform_vector_search(
    es_client, index_name, user_question, search_type="knn", k=5
)

for doc in results:
    print(
        f"Question: {doc['question']}, \n Text: {doc['text']}, \n doc_id: {doc['id']}, \n score: {doc['score']}\n\n"
    )

# The id with the highest score doc_id: ee58a693,


# Calculate hit rate for Elasticsearch on all the questions in ground_truth
def hit_rate_elasticsearch(
    es_client, index_name, user_question, num_results=5, search_type="cosine"
):
    question = user_question["question"]
    doc_id = user_question["document"]
    results = perform_vector_search(
        es_client, index_name, question, search_type=search_type, k=num_results
    )
    hit = any(result["id"] == doc_id for result in results)
    return hit


# Calculate hit rates for Elasticsearch using cosine similarity
hit_rates_elastic_cosine = [
    hit_rate_elasticsearch(
        es_client, index_name, doc, num_results=5, search_type="cosine"
    )
    for doc in tqdm(ground_truth)
]
final_hit_rate_elastic_cosine = np.mean(hit_rates_elastic_cosine)
print(
    f"Final hit rate for Elasticsearch (cosine similarity): {final_hit_rate_elastic_cosine}"
)

# Calculate hit rates for Elasticsearch using kNN
hit_rates_elastic_knn = [
    hit_rate_elasticsearch(es_client, index_name, doc, num_results=5, search_type="knn")
    for doc in tqdm(ground_truth)
]
final_hit_rate_elastic_knn = np.mean(hit_rates_elastic_knn)
print(f"Final hit rate for Elasticsearch (kNN): {final_hit_rate_elastic_knn}")
