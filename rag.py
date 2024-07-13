from QdrantU import QdrantU
from Processing import TextEmbedder
import cohere
from Helpers import generate_prompt, get_docs_by_indices, query_rewriter
import os
import dotenv

dotenv.load_dotenv()

def run_rag(query, history=None):
    """
    This function performs Retrieval-Augmented Generation (RAG) for a given query.

    Args:
        query (str): The user's original query.
        history (list, optional): A list of previous prompts and responses (for potential future use). Defaults to None.

    Returns:
        str: The generated prompt based on the retrieved documents and query.
    """

    # Text embedding model for document representation
    embedding_model = TextEmbedder()

    # QdrantU client for document search
    uploader = QdrantU(collection_name='News_Articles_Source')

    # Attempt query rewriting
    try:
        query = query_rewriter(query)
        print("Query after rewriting: ", query)
    except Exception as e:
        print("Error in query rewriting:", e)
        pass 


    search_results = uploader.search(query, embedding_model, limit=1000)

    docs = list(set([result.payload['content'] for result in search_results]))

    apiKey = os.getenv("cohere_api_key")

    try:
        co = cohere.Client(apiKey)

        # Rerank documents using Cohere's rerank-english-v3.0 model
        rerank_docs = co.rerank(
            query=query,
            documents=docs,
            top_n=2, 
            model="rerank-english-v3.0"
        )

        indices = [result.index for result in rerank_docs.results]

        documents = get_docs_by_indices(docs, indices)
    except Exception as e:
        print("Error in reranking:", e)

        documents = docs[:2] 

    prompt = generate_prompt(documents, query, history)

    return prompt