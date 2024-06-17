from QdrantU import QdrantU
from Processing import TextEmbedder
import cohere
from Helpers import generate_prompt, get_docs_by_indices, query_rewriter

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

    # Attempt query rewriting (optional, handle potential errors)
    try:
        query = query_rewriter(query)
        print("Query after rewriting: ", query)
    except Exception as e:
        print("Error in query rewriting:", e)
        pass  # Continue processing even if rewriting fails

    # Search for relevant documents using the embedding model and limit results
    search_results = uploader.search(query, embedding_model, limit=1000)

    # Extract document content from search results and remove duplicates
    docs = list(set([result.payload['content'] for result in search_results]))

    # Cohere API key for reranking (replace with your own key)
    apiKey = 'Q21IIAUkTtt1jk9WUgJg0XiCvaU2K73cFbq0djhM'
    co = cohere.Client(apiKey)

    # Rerank documents using Cohere's rerank-english-v3.0 model
    rerank_docs = co.rerank(
        query=query,
        documents=docs,
        top_n=2,  # Select the top 2 reranked documents
        model="rerank-english-v3.0"
    )

    # Extract document indices from the reranked results
    indices = [result.index for result in rerank_docs.results]

    # Retrieve the full content of the reranked documents
    documents = get_docs_by_indices(docs, indices)

    # Generate a prompt based on the retrieved documents, query, and history (if provided)
    prompt = generate_prompt(documents, query, history)

    return prompt