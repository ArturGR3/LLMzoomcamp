import json
import sys 

def modify_docs(doc) -> list: 
    """ 
    Load the documents from the JSON file and append the course name to each document in the 'documents' list.
    Return the documents list.
    """
    # Load the documents from the JSON file
    with open(doc, 'rt') as f_in:
        docs_raw = json.load(f_in)
    
    documents = []

    # Append the course name to each document in the 'documents' list
    for course_dict in docs_raw:
        for doc in course_dict['documents']:
            doc['course'] = course_dict['course']
            documents.append(doc)
    return documents


# Make this function executable from the command line:
if __name__ == '__main__':
    modify_docs()