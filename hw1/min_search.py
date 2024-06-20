import minsearch
import json 
import modify_doc

def minisearch(query, document, num_results, course) -> minsearch.Index:
    # Create an instance of the Index class
    Index = minsearch.Index(text_fields=["question", "text", "section"],
                            keyword_fields=["course"])
    Index.fit(document)
    
    boost = {'question' : 3.0} # question field is 3 times more important (default is 1, less importants is between 0 and 1)
    
    search_result = Index.search(
        query = query,
        boost_dict = boost,
        num_results = num_results, # number of results returned 
        filter_dict = {'course':course} 
    )
    
    return search_result
