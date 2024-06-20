from elasticsearch import Elasticsearch
from tqdm import tqdm
from modify_doc import modify_docs

# documents = modify_docs(doc = 'documents.json')

def start_elasticsearch(documents) -> tuple[Elasticsearch, str]:
    """
    Start Elasticsearch, create an index, and index the documents.
    """
    # Start Elasticsearch
    es_client = Elasticsearch('http://localhost:9200')
    
    # Check if the port is reachable
    if not es_client.ping():
        raise ValueError("Connection failed")

    # Create an index in Elasticsearch
    index_settings = {
        "settings": {"number_of_shards": 1, "number_of_replicas": 0},
        "mappings": {
            "properties": {
                "text": {"type": "text"},
                "question": {"type": "text"},
                "section": {"type": "text"},
                "course": {"type": "keyword"}
            }
        }
    }
    index_name = "course-question"
    # If the index already exists, delete it
    if es_client.indices.exists(index=index_name):
        es_client.indices.delete(index=index_name)
    
    # Create the index    
    es_client.indices.create(index=index_name, body=index_settings)

    # Index the documents in Elasticsearch
    for doc in tqdm(documents):
        es_client.index(index=index_name, body=doc)
    
    return es_client, index_name
    
def elastic_search(query, document, course, num_results) -> list:
    """
    Search the question in the Elasticsearch index and return the search result.
    
    Parameters:
    query (str): The search query to be used for searching in the Elasticsearch index.
    
    Returns:
    list: A list of search results matching the query.
    """
    # Define the search query for Elasticsearch
    search_query = {
        "size": num_results,  # Limit the number of search results to 5
        "query": {
            "bool": {
                "must": {
                    "multi_match": {
                        "query": query,  # The search query
                        "fields": ["question^4", "text"],  # Fields to search in, with "question" field given 4 times more importance
                        "type": "best_fields"  # Use the best matching field
                    }
                },
                "filter": {
                    "term": {
                        "course": course  # Filter the search results by the "course" field value
                    }
                }
            }
        }
    }
    
    es_client, index_name = start_elasticsearch(document)
    
    response = es_client.search(index=index_name, body=search_query)
    result_docs = []
    
    for i,hit in enumerate(response['hits']['hits']):
        print(f"Top {i+1} Score for {course}: {hit['_score']}")
        result_docs.append(hit['_source'])
        
    return result_docs


