import re
import time
import requests
import json
import spacy
import string
import torch
from textblob import TextBlob
from transformers import BertTokenizer, BertModel, AutoTokenizer
from sentence_transformers import SentenceTransformer, util

# Load BERT tokenizer and model
model_name = "bert-base-uncased"
Bert_tokenizer = AutoTokenizer.from_pretrained(model_name)
Bert_model = BertModel.from_pretrained(model_name)

# Load SentenceTransformer model
sentence_transformer_model = SentenceTransformer("all-mpnet-base-v2")

def generate_prompt(context, question, history=None):
    context = ". ".join(context)
    print(context)
    print("calculating the similarity...")
    if validate_revised_query(context, question, threshold=0.4):
        prompt_context = context
    else:
        prompt_context = "No context provided, Response based on the question only."
    prompt = f"""
    <s><<SYS>> You are a helpful, respectful and honest assistant. Always answer as helpfully as possible based on the context, while being safe. Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content, and dont mention that you used the provided context.do not add any Additional Questions<</SYS>>

    Context \n :
    {prompt_context}

    [INST] {question} [/INST]

    Response:
    """
 
    return prompt

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

def question_answering(question):
    headers = {"Authorization": "Bearer eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJ1c2VyX2lkIjoiMmYzMDE3MTEtOTJmNy00ZDU3LTg4N2MtNjU2MmE5MTU5MWZhIiwidHlwZSI6ImFwaV90b2tlbiJ9.vWvooRwxmr-uY1c61V97uugyDGpXmZGjX8oCFWKCUeM"}
    url = "https://api.edenai.run/v2/text/question_answer"
    payload = {
        "providers": "openai",
        "texts": [
            "Linux is a family of open-source Unix-like operating systems based on the Linux kernel, an operating system kernel first released on September 17, 1991, by Linus Torvalds.",
            "Just like Windows, iOS, and Mac OS, Linux is an operating system."
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

def normalize_text(s):
    nlp = spacy.load("en_core_web_sm")
    def remove_stop(text):
        return " ".join([word for word in text.split() if not nlp.vocab[word].is_stop])
    def lemma(text):
        return " ".join([word.lemma_ for word in nlp(text)])
    def white_space_fix(text):
        return " ".join(text.split())
    def remove_punc(text):
        exclude = set(string.punctuation)
        return "".join(ch for ch in text if ch not in exclude)
    def lower(text):
        return text.lower()
    return white_space_fix(lemma(remove_stop(remove_punc(lower(s)))))

def get_relevance_docs(documents_score, threshold):
    relevance_scores = []
    for score in documents_score:
        if score >= threshold:
            relevance_scores.append(1)
        else:
            relevance_scores.append(0)
    return relevance_scores

def get_docs_by_indices(docs, indices):
    return [docs[index] for index in indices]

def split_text(text):
    return text.split(":")[1].strip() if ":" in text else text 

def query_rewriter(original_query):
    if not needs_rewriting(original_query):
        return original_query
    try:
        headers = {"Authorization": "Bearer eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJ1c2VyX2lkIjoiMmYzMDE3MTEtOTJmNy00ZDU3LTg4N2MtNjU2MmE5MTU5MWZhIiwidHlwZSI6ImFwaV90b2tlbiJ9.vWvooRwxmr-uY1c61V97uugyDGpXmZGjX8oCFWKCUeM"}
        url = "https://api.edenai.run/v2/text/code_generation"
        payload = {
            "providers": "openai",
            "prompt": "",
            "model": "gpt-3.5-turbo",
            "instruction": f"""You are an expert in document retrieval and search optimization.
            Your task is to rewrite the following query to enhance its relevance and usefulness for retrieving accurate and 
            comprehensive information from a database or search engine. Ensure the rewritten query is clear, specific,
            and free of ambiguities. Here are a few examples:
                Original Query: who is Joe Biden? Rewritten Query: Provide detailed information about Joe Biden, including his political career, achievements, current position, and a history of his personal and professional life?
            Original Query: {original_query}""",
            "temperature": 0.6,
            "max_tokens": 512,
            "fallback_providers": "['openai']"
        }
        response = requests.post(url, json=payload, headers=headers, timeout=20)
        result = json.loads(response.text)
    except:
        print("Error in API call")
        return original_query
    try:
        rewrited_query = split_text(result['openai']['generated_text'])
    except:
        return original_query
    if validate_revised_query(original_query, rewrited_query):
        return result['openai']['generated_text']
    else:
        return original_query

def score_query(query):
    score = 0
    salutations = ['hi', 'hello', 'hey', 'dear', 'greetings', 'good morning', 'good afternoon', 'good evening', 'good night', 'good day', 'howdy', 'what\'s up', 'sup', 'yo', 'hiya', 'hi there', 'hello there', 'hey there', 'hiya there', 'howdy there', 'what\'s up there', 'sup there', 'yo there']
    if any(salutation in query.lower() for salutation in salutations):
        return score
    if len(query.split()) < 3 or len(query.split()) > 15:
        score += 1
    question_pattern = r'\b(who|what|where|when|why|how|which|whom|whose)\b'
    if not re.search(question_pattern, query.lower()):
        score += 2
    blob = TextBlob(query)
    if len(blob.correct().words) != len(blob.words):
        score += 1
    if blob.correct() != blob:
        score += 1
    ambiguous_terms = ['these', 'such', 'something', 'one', 'those', 'whatchamacallit', 'doohickey', 'whosit', 'matter', 'aspect', 'case', 'concept', 'issue', 'point', 'area', 'facet', 'data', 'information', 'people', 'stuff', 'business', 'thingy', 'whatnot', 'deal']
    if any(term in query.lower() for term in ambiguous_terms):
        score += 1
    if len(set(query.split())) < len(query.split()) * 0.5:
        score += 1
    return score

def needs_rewriting(query, threshold=2):
    print(f"Scoring query: '{query}'")
    score = score_query(query)
    print(f"Score: {score}")
    return score > threshold

def get_bert_embeddings(sentence, use_sentence_transformer=False):
    if use_sentence_transformer:
        sentence_embedding = sentence_transformer_model.encode(sentence, convert_to_tensor=True)
    else:
        inputs = Bert_tokenizer(sentence, return_tensors="pt", padding=True, truncation=True)
        with torch.no_grad():
            outputs = Bert_model(**inputs)
            sentence_embedding = outputs.last_hidden_state[:, 0, :]
    return sentence_embedding

def compute_bert_similarity(sentence1, sentence2, use_sentence_transformer=False):
    if use_sentence_transformer:
        sentence_embeddings = sentence_transformer_model.encode([sentence1, sentence2], convert_to_tensor=True)
    else:
        sentence1_embedding = get_bert_embeddings(sentence1, use_sentence_transformer)
        sentence2_embedding = get_bert_embeddings(sentence2, use_sentence_transformer)
        sentence_embeddings = [sentence1_embedding, sentence2_embedding]
    cosine_similarity = util.pytorch_cos_sim(sentence_embeddings[0], sentence_embeddings[1]).item()
    return cosine_similarity

def validate_revised_query(original_query, revised_query, threshold=0.6):
    similarity = compute_bert_similarity(original_query, revised_query, use_sentence_transformer=True)
    print(f"Similarity: {similarity}")
    return similarity > threshold