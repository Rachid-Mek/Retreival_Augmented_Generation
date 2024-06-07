from QdrantU import QdrantU
from Processing import TextEmbedder 
import cohere
from Helpers import generate_prompt, llama, get_docs_by_indices , query_rewriter


def run_rag(query, history=None):
    embedding_model = TextEmbedder()
    uploader = QdrantU(collection_name='News_Articles_Source')
    try: 
        query = query_rewriter(query) 
        print("Query after rewriting: ", query)
    except:
        print("Error in query rewriting")
        pass
    search_results = uploader.search(query, embedding_model, limit=1000)
    docs = list(set([result.payload['content'] for result in search_results]))

    apiKey = 'Q21IIAUkTtt1jk9WUgJg0XiCvaU2K73cFbq0djhM' # API key for Cohere
    co = cohere.Client(apiKey)
    rerank_docs = co.rerank(
        query=query, documents=docs, top_n=2, model="rerank-english-v3.0"
    )

    indices = [result.index for result in rerank_docs.results]
    documents = get_docs_by_indices(docs, indices)
    prompt = generate_prompt(documents, query, history)

    return  prompt