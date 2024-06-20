
from elastic_search import elastic_search
from modify_doc import modify_docs

# documents = modify_docs(doc = 'documents.json')
# search_result = elastic_search(
#     query = 'How do I execute a command in a running docker container?', 
#     document = documents, 
#     course = 'machine-learning-zoomcamp', 
#     num_results = 3)

def build_prompt(query, search_result) -> str:
    """
    Build a prompt for the OpenAI API based on the search result.
    """
    prompt_template = """
You're a course teaching assistant. Answer the QUESTION based on the CONTEXT from the FAQ database.
Use only the facts from the CONTEXT when answering the QUESTION.

QUESTION: {question}

CONTEXT:
{context}
    """.strip()
    # Check if the search result has section, question and text fields
    if not all(doc.get('section') and doc.get('question') and doc.get('text') for doc in search_result):
        return "The context doesn't provide enough information to answer the question."

    context = ""

    # Build the context for the prompt separate each other by \n\n
    for doc in search_result:
        context += f"Question: {doc['question']}\nAnswer: {doc['text']}\n\n"
        

    prompt = prompt_template.format(question=query, context=context.strip())
    

    return prompt

# q = 'How do I execute a command in a running docker container?'

# t = build_prompt(q, search_result)

# print(len(t))