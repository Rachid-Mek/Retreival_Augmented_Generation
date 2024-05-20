import re
import time
import requests
import json
import spacy
import string

from textblob import TextBlob
import torch
from transformers import BertTokenizer, BertModel
from sklearn.metrics.pairwise import cosine_similarity

Bert_tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
Bert_model = BertModel.from_pretrained('bert-base-uncased')

def generate_prompt(context, question, history=None):
        
    # history_summary = ""
    # if history:
    #     for user_query, bot_response in history[-3:]:  
    #         history_summary += f"User: {user_query}\n    Assistant: {bot_response}\n"
    if context:
        prompt_context = context
    else:
        prompt_context = "No context provided."
    prompt = f"""
    <s>[INST] <<SYS>> You are a helpful, respectful and honest assistant. Always answer as helpfully as possible based on the context, while being safe.  Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content, and dont mention that you used the provided context .<</SYS>>

    Context \n :
    {prompt_context}
 
    [INST] {question} [/INST]

     Response:
    """

    return prompt

# ==============================================================================================================================================
def llama(prompt):
    url = "https://api.edenai.run/v2/text/generation"
    payload = {
        "providers": "meta/llama2-13b-chat-v1",
        "response_as_dict": True,
        "attributes_as_list": False,
        "show_original_response": False,
        "temperature": 0,
        "max_tokens": 256,
        "text": prompt
    }
    headers = {
        "accept": "application/json",
        "content-type": "application/json",
        "authorization": "Bearer eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJ1c2VyX2lkIjoiMmYzMDE3MTEtOTJmNy00ZDU3LTg4N2MtNjU2MmE5MTU5MWZhIiwidHlwZSI6ImFwaV90b2tlbiJ9.vWvooRwxmr-uY1c61V97uugyDGpXmZGjX8oCFWKCUeM"
    }

    response = requests.post(url, json=payload, headers=headers)
    result = response.json()
    return result['meta/llama2-13b-chat-v1']['generated_text']
# ==============================================================================================================================================

def question_answering(question):
  """
  Sends a question answering request to the EdenAI API.

  Args:
      question: The question to be answered.

  Returns:
      The answer provided by the LLM model (string),
      or None if an error occurs.
  """
  headers = {"Authorization": "Bearer eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJ1c2VyX2lkIjoiMmYzMDE3MTEtOTJmNy00ZDU3LTg4N2MtNjU2MmE5MTU5MWZhIiwidHlwZSI6ImFwaV90b2tlbiJ9.vWvooRwxmr-uY1c61V97uugyDGpXmZGjX8oCFWKCUeM"}

  url = "https://api.edenai.run/v2/text/question_answer"
  payload = {
      "providers": "openai",
      "texts": [
          "Linux is a family of open-source Unix-like operating systems based on the Linux kernel, an operating system kernel first released on September 17, 1991, by Linus Torvalds.",
          "Just like Windows, iOS, and Mac OS, Linux is an operating system. "
      ],
      'question': question,
      "examples": [["What is human life expectancy in the United States?", "78 years."]],
      "fallback_providers": ""
  }

  try:
    response = requests.post(url, json=payload, headers=headers)
    result = json.loads(response.text)
    return result['openai']['answers'] if result['openai']['answers'] else None
  except Exception as e:
    print(f"Error communicating with LLM model: {e}")
    return None
  
# ==============================================================================================================================================
def normalize_text(s):
    """Removing stopwords and punctuation, and standardizing whitespace are all typical text processing steps."""
    
    nlp = spacy.load("en_core_web_sm")
    def remove_stop(text):
        return " ".join([word for word in text.split() if not nlp.vocab[word].is_stop])
    
    def lemma(text):
        return " ".join([word.lemma_ for word in nlp(text)])
    
    def white_space_fix(text):
        return " ".join(text.split()) # this function removes leading and trailing whitespaces and condenses all other whitespaces to a single space

    def remove_punc(text):
        exclude = set(string.punctuation)
        return "".join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(lemma(remove_stop(remove_punc(lower(s)))))

# ==============================================================================================================================================
def get_relevance_docs(documents_score, threshold):
    """
    Calculate relevance scores for the retrieved documents based on their relevance to the correct answer.

    Parameters:
        documents_score (list): List of scores for the retrieved documents.
        threshold (float): Threshold value to determine relevance.

    Returns:
        list: List of relevance scores for the retrieved documents.
    """
    relevance_scores = []
    for score in documents_score:
        if score >= threshold:
            relevance_scores.append(1)  # Relevant document
        else:
            relevance_scores.append(0)  # Non-relevant document
    return relevance_scores

# ==============================================================================================================================================
def get_docs_by_indices(docs, indices):
    """
    Retrieve document contexts from a list of indexed documents based on provided indices.

    Args:
    - docs (list): List of documents.
    - indices (list): List of indices corresponding to the desired documents.

    Returns:
    - list: List of document contexts corresponding to the provided indices.
    """
    return [docs[index] for index in indices]

# ==============================================================================================================================================
 
def split_text(text):
    return text.split(":")[1].strip() if ":" in text else text 
def query_rewriter(original_query):
    timing=0
    if not needs_rewriting(original_query):
        return original_query
    try:
        # headers = {"Authorization": "Bearer eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJ1c2VyX2lkIjoiMTA5MTBlYTktOWYwOC00N2E2LTg3MDktOTlhODExZjkwZDA2IiwidHlwZSI6ImFwaV90b2tlbiJ9._wiFq518MhMRvG8waWbg_7Eogf50isgyzqh3e2ypvOU"}
        headers = {"Authorization": "Bearer eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJ1c2VyX2lkIjoiMmYzMDE3MTEtOTJmNy00ZDU3LTg4N2MtNjU2MmE5MTU5MWZhIiwidHlwZSI6ImFwaV90b2tlbiJ9.vWvooRwxmr-uY1c61V97uugyDGpXmZGjX8oCFWKCUeM"}
        url = "https://api.edenai.run/v2/text/code_generation"
        payload = {
            "providers": "openai",
            # "instruction": "You are an expert at world knowledge. Your task is to step back and paraphrase a question to a more generic step-back question, which is easier to answer. Here are a few examples:Original Question: Which position did Knox Cunningham hold from May 1955 to Apr 1956? Stepback Question: Which positions have Knox Cunning- ham held in his career? , Now this Question: Who was the spouse of Anna Karina from 1968 to 1974?",
            "prompt": "",
            "model": "gpt-3.5-turbo",
            "instruction": f"""You are an expert in document retrieval and search optimization.
            Your task is to rewrite the following query to enhance its relevance and usefulness for retrieving accurate and 
            comprehensive information from a database or search engine. Ensure the rewritten query is clear, specific,
            and free of ambiguities. Here are a few examples:
                Original Query: who is Joe Biden? Rewritten Query: Provide detailed information about Joe Biden, including his political career, achievements, current position, and a history of his personal and professional life?                
            Original Query: {original_query}"""  ,
            "temperature": 0.6,
            "max_tokens": 512,
            "fallback_providers": " ['openai']"
        }

        response = requests.post(url, json=payload, headers=headers , timeout=20)
        print("response", response)
        result = json.loads(response.text)
        print("result", result)
        
    except:
        print("Error in API call")
        return original_query
    try:   
         print("query revised", split_text(result['openai']['generated_text']))
         rewrited_query = split_text(result['openai']['generated_text'])
    except:
        return original_query
    
    if validate_revised_query(original_query ,rewrited_query):
        return result['openai']['generated_text']
    else:
        return original_query


def score_query(query):
    score = 0
    salutations = ['hi', 'hello', 'hey', 'dear', 'greetings', 'good morning', 'good afternoon', 'good evening', 'good night', 'good day', 'howdy', 'what\'s up', 'sup', 'yo', 'hiya', 'hi there', 'hello there', 'hey there', 'hiya there', 'howdy there', 'what\'s up there', 'sup there', 'yo there', ]
    if any(salutation in query.lower() for salutation in salutations):
        return score
    # Criterion 1: Length of Query
    if len(query.split()) < 3 or len(query.split()) > 15:
        score += 1
    question_pattern = r'\b(who|what|where|when|why|how|which|whom|whose)\b'
    if  not re.search(question_pattern, query.lower()) :
        score += 2

    # Criterion 2: Spelling Errors
    blob = TextBlob(query)
    if len(blob.correct().words) != len(blob.words):
        score += 1
 
    # Criterion 3: Grammar Issues (simplified)
    if blob.correct() != blob:
        score += 1

    # Criterion 4: Ambiguity (simplified example)
    ambiguous_terms = ['these', 'such', 'something', 'one', 'those', 'whatchamacallit', 'doohickey', 'whosit', 'matter', 'aspect', 'case', 'concept', 'issue', 'point', 'area', 'facet''data', 'information', 'people', 'stuff', 'business', 'thingy', 'whatnot', 'deal']
    if any(term in query.lower() for term in ambiguous_terms):
        score += 1
    
    # Criterion 5: Complexity
    if len(set(query.split())) < len(query.split()) * 0.5:
        score += 1
    return score

def needs_rewriting(query, threshold=2):
    print(f"Scoring query: '{query}'")
    score = score_query(query)
    print(f"Score: {score}")
    return score > threshold

def get_bert_embeddings(text):
    inputs = Bert_tokenizer(text, return_tensors='pt', truncation=True, padding=True, max_length=512)
    with torch.no_grad():
        outputs = Bert_model(**inputs)
    embeddings = outputs.last_hidden_state.mean(dim=1).squeeze()
    return embeddings

def compute_semantic_similarity(text1, text2):
    embeddings1 = get_bert_embeddings(text1)
    embeddings2 = get_bert_embeddings(text2)
    similarity = cosine_similarity(embeddings1.reshape(1, -1), embeddings2.reshape(1, -1))
    return similarity[0][0]

def validate_revised_query(original_query, revised_query, threshold=0.6):
    similarity = compute_semantic_similarity(original_query, revised_query)
    print(f"Similarity: {similarity}")
    return similarity > threshold
