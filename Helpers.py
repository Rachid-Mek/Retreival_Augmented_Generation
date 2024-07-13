import re
import requests
import json
import spacy
import string
import torch
from textblob import TextBlob
from transformers import BertModel, AutoTokenizer
from sentence_transformers import SentenceTransformer, util

# Load BERT tokenizer and model
model_name = "bert-base-uncased"
Bert_tokenizer = AutoTokenizer.from_pretrained(model_name)
Bert_model = BertModel.from_pretrained(model_name)

# Load SentenceTransformer model
sentence_transformer_model = SentenceTransformer("all-mpnet-base-v2")

# -------------------------------------------------------------------------------------------------------

# def generate_prompt(context, question, history=None):
#     history_summary = ""
#     if history:
#         for entry in history[-3:]:  # Limit to the last 3 entries for brevity
#             user_query, bot_response = entry["role"], entry["content"]
#             history_summary += f"User: {user_query}\nAssistant: {bot_response}\n"
    
#     context = ". ".join(context)
#     print(context)
#     print("Calculating the similarity...")
    
#     if validate_revised_query(context, question, threshold=0.4):
#         prompt_context = context
#     else:
#         prompt_context = "No context provided. Response based on the question only."
    
#     prompt = f"""
#     <|start_header_id|>system<|end_header_id|> You are a helpful, respectful, and honest assistant. Always answer as helpfully as possible based on the context, while being safe. Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. Do not mention that you used the provided context. Do not add any additional questions.
#     Conversation History:
#     {history_summary}
#     Context:
#     {prompt_context} <|eot_id|>

#     <|start_header_id|>user<|end_header_id|> This is the question:
#     {question} <|eot_id|>
#     Response:
#     """
 
#     return prompt

def generate_prompt(context, question, history=None):
    """
    This function generates a prompt for a large language model (LLM) based on context, question, and history.
    Args:
        context (list): A list of strings representing the contextual information.
        question (str): The user's question.
        history (list, optional): A list of previous prompts and responses (for potential future use). Defaults to None.
    Returns:
        str: The generated prompt formatted for the LLM.
    """

    history_summary = ""
    if history is not []:
        # Limit summary to the last 3 entries for conciseness
        for entry in history[-3:]:
            print("entry" , entry)
            role, content = entry["role"], entry["content"]
            if role == "user":
                history_summary += f"User: {content}\n"
            elif role == "system":
                history_summary += f"Assistant: {content}\n"

    # Combine context sentences into a single string
    context = ". ".join(context)
    # print(context)
    print("Calculating the similarity between context and question...")

    # Check if context and question are similar enough to avoid unnecessary context usage
    if validate_revised_query(context, question, threshold=0.4):
        prompt_context = context
    else:
        prompt_context = "No context provided. Response based on the question only."
    if history_summary == "":
            prompt = f"""
              You are a helpful, respectful, and honest assistant. Always answer as helpfully as possible based on the context, while being safe. Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. Do not mention that you used the provided context. Do not add any additional questions.
              Context:
              {prompt_context}
              User: 
              {question}
              Assistant:
              """
    else:
        prompt = f"""
              You are a helpful, respectful, and honest assistant. Always answer as helpfully as possible based on the context, while being safe. Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. Do not mention that you used the provided context. Do not add any additional questions.
              Conversation History:
                    {history_summary}
              Context:
                    {prompt_context}
              User: 
                    {question}
              Assistant:
              """
    # print("Prompt : ", prompt)
    return prompt


# -------------------------------------------------------------------------------------------------------

def llama(prompt):
    """
    This function sends a prompt to the Llama LLM API and returns the generated text.

    Args:
        prompt (str): The prompt to be processed by the LLM.

    Returns:
        str: The generated text from the Llama LLM.
    """
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
#-------------------------------------------------------------------------------------------------------
def question_answering(question):
  """
  This function sends a question and context to the OpenAI LLM API for question answering.

  Args:
      question (str): The user's question.

  Returns:
      list or None: A list of answer strings if successful, None otherwise.
  """

  # Replace with your actual OpenAI API key
  headers = {"Authorization": "Bearer eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJ1c2VyX2lkIjoiMmYzMDE3MTEtOTJmNy00ZDU3LTg4N2MtNjU2MmE5MTU5MWZhIiwidHlwZSI6ImFwaV90b2tlbiJ9.vWvooRwxmr-uY1c61V97uugyDGpXmZGjX8oCFWKCUeM"}

  # URL for OpenAI question answering endpoint
  url = "https://api.edenai.run/v2/text/question_answer"

  # Payload for the request containing question and context
  payload = {
    "providers": "openai",  # Specify OpenAI as the provider
    "texts": [ # List of text snippets for context
      "Linux is a family of open-source Unix-like operating systems based on the Linux kernel, an operating system kernel first released on September 17, 1991, by Linus Torvalds.",
      "Just like Windows, iOS, and Mac OS, Linux is an operating system."
    ],
    "question": question,  # User's question to be answered
    "examples": [  # Optional: Example question-answer pairs to guide the model
      ["What is human life expectancy in the United States?", "78 years."]
    ],
    "fallback_providers": ""  # Optional: Providers to use if OpenAI fails
  }

  try:
    # Send POST request with JSON payload and headers
    response = requests.post(url, json=payload, headers=headers)
    
    # Parse the JSON response
    result = json.loads(response.text)

    # Extract answers from the OpenAI response if successful, otherwise return None
    return result['openai']['answers'] if result['openai']['answers'] else None

  except Exception as e:
    # Handle potential exceptions during communication with the LLM model
    print(f"Error communicating with LLM model: {e}")
    return None
#-------------------------------------------------------------------------------------------------------
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
#-------------------------------------------------------------------------------------------------------
def get_relevance_docs(documents_score, threshold):
    '''
    this function takes a list of documents' scores and a threshold value and returns a 
    list of relevance scores

    documents_score: list of documents' scores
    threshold: float value to determine the relevance of the documents
    '''
    relevance_scores = []
    for score in documents_score:
        if score >= threshold:
            relevance_scores.append(1)
        else:
            relevance_scores.append(0)
    return relevance_scores
#-------------------------------------------------------------------------------------------------------
def get_docs_by_indices(docs, indices):
    '''
    this function takes a list of documents and a list of indices and returns a list of documents
    '''
    return [docs[index] for index in indices]
#-------------------------------------------------------------------------------------------------------
def split_text(text):
    return text.split(":")[1].strip() if ":" in text else text 
#-------------------------------------------------------------------------------------------------------
def query_rewriter(original_query):
  """
  This function rewrites a user's query to improve its effectiveness for information retrieval.

  Args:
      original_query (str): The user's original query.

  Returns:
      str: The rewritten query (or the original query if no rewriting is needed).
  """

  # Check if the query needs rewriting based on some criteria (not implemented here)
  if not needs_rewriting(original_query):
    return original_query

  try:
    # Authorization header with placeholder (replace with your actual API key)
    headers = {"Authorization": "Bearer eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJ1c2VyX2lkIjoiMmYzMDE3MTEtOTJmNy00ZDU3LTg4N2MtNjU2MmE5MTU5MWZhIiwidHlwZSI6ImFwaV90b2tlbiJ9.vWvooRwxmr-uY1c61V97uugyDGpXmZGjX8oCFWKCUeM"}

    # URL for the OpenAI code generation endpoint
    url = "https://api.edenai.run/v2/text/code_generation"

    # Payload for the request
    payload = {
      "providers": "openai",  # Specify OpenAI as the provider
      "prompt": "",  # Placeholder, will be filled later
      "model": "gpt-3.5-turbo",  # Specify the OpenAI model to use
      "instruction": f"""
      You are an expert in document retrieval and search optimization.
      Your task is to rewrite the following query to enhance its relevance and usefulness for retrieving accurate and 
      comprehensive information from a database or search engine. Ensure the rewritten query is clear, specific,
      and free of ambiguities. Here are a few examples:
          Original Query: who is Joe Biden? Rewritten Query: Provide detailed information about Joe Biden, including his political career, achievements, current position, and a history of his personal and professional life?
      Original Query: {original_query}""",  # Include the original query in the prompt
      "temperature": 0.6,  # Control the randomness of the generated text (0.6 is a medium value)
      "max_tokens": 512,  # Maximum number of tokens to generate
      "fallback_providers": "['openai']"  # Alternative provider if OpenAI fails
    }

    # Send POST request with JSON payload and headers
    response = requests.post(url, json=payload, headers=headers, timeout=20)

    # Parse the JSON response
    result = json.loads(response.text)

  except Exception as e:
    # Handle errors during API call
    print("Error in API call:", e)
    return original_query

  try:
    # Extract the rewritten query from the response (assuming a specific format)
    rewritten_query = split_text(result['openai']['generated_text'])  # Implement split_text function
  except Exception as e:
    # Handle errors during parsing the response
    print("Error parsing response:", e)
    return original_query

  # Check if the rewritten query is actually different and useful (not implemented here)
  if validate_revised_query(original_query, rewritten_query):
    return result['openai']['generated_text']  # Return the rewritten query
  else:
    return original_query  # Return the original query if rewriting was not helpful

#-------------------------------------------------------------------------------------------------------
def score_query(query):
  """
  This function assigns a score to a user's query based on its characteristics.

  Args:
      query (str): The user's query.

  Returns:
      int: A score indicating the potential complexity of understanding the query. Higher scores suggest a more complex query.
  """

  score = 0

  # Check for greetings (these don't require complex processing)
  salutations = ['hi', 'hello', 'hey', 'dear', 'greetings', 'good morning', 'good afternoon', 'good evening', 'good night', 'good day', 'howdy', 'what\'s up', 'sup', 'yo', 'hiya', 'hi there', 'hello there', 'hey there', 'hiya there', 'howdy there', 'what\'s up there', 'sup there', 'yo there']
  if any(salutation in query.lower() for salutation in salutations):
    # Greetings get a score of 0 (considered simple)
    return score

  # Penalize very short or long queries (might be incomplete or unfocused)
  if len(query.split()) < 3 or len(query.split()) > 15:
    score += 1  # Short or long queries get a penalty

  # Check for question words (indicates an information-seeking intent)
  question_pattern = r'\b(who|what|where|when|why|how|which|whom|whose)\b'
  if not re.search(question_pattern, query.lower()):
    score += 2  # Lack of question words gets a higher penalty

  # Check for spelling errors and grammatical mistakes (might be harder to understand)
  blob = TextBlob(query)
  if len(blob.correct().words) != len(blob.words):
    score += 1  # Spelling errors get a penalty
  if blob.correct() != blob:
    score += 1  # Grammatical mistakes get a penalty

  # Check for ambiguous terms (might be unclear what the user is referring to)
  ambiguous_terms = ['these', 'such', 'something', 'one', 'those', 'whatchamacallit', 'doohickey', 'whosit', 'matter', 'aspect', 'case', 'concept', 'issue', 'point', 'area', 'facet', 'data', 'information', 'people', 'stuff', 'business', 'thingy', 'whatnot', 'deal']
  if any(term in query.lower() for term in ambiguous_terms):
    score += 1  # Ambiguous terms get a penalty

  # Check for repetitive words (might indicate a lack of clarity)
  if len(set(query.split())) < len(query.split()) * 0.5:
    score += 1  # Repetitive words get a penalty

  return score
#-------------------------------------------------------------------------------------------------------
def needs_rewriting(query, threshold=2):
    print(f"Scoring query: '{query}'")
    score = score_query(query)
    print(f"Score: {score}")
    return score > threshold
#-------------------------------------------------------------------------------------------------------
def get_bert_embeddings(sentence, use_sentence_transformer=False):
  """
  This function generates sentence embeddings using a pre-trained BERT model or SentenceTransformer model.

  Args:
      sentence (str): The sentence to generate the embedding for.
      use_sentence_transformer (bool, optional): Flag to choose between BERT or SentenceTransformer. Defaults to False (BERT).

  Returns:
      torch.Tensor: The sentence embedding as a PyTorch tensor.
  """

  if use_sentence_transformer:
    # Use SentenceTransformer model for sentence embedding
    sentence_embedding = sentence_transformer_model.encode(sentence, convert_to_tensor=True)
  else:
    # Use BERT model for sentence embedding
    inputs = Bert_tokenizer(sentence, return_tensors="pt", padding=True, truncation=True)
    with torch.no_grad():
      # Disable gradient calculation for efficiency during embedding generation
      outputs = Bert_model(**inputs)
      sentence_embedding = outputs.last_hidden_state[:, 0, :]

  return sentence_embedding
#-------------------------------------------------------------------------------------------------------
def compute_bert_similarity(sentence1, sentence2, use_sentence_transformer=False):
  """
  This function computes the cosine similarity between two sentences using sentence embeddings generated from a pre-trained BERT model or SentenceTransformer model.

  Args:
      sentence1 (str): The first sentence.
      sentence2 (str): The second sentence.
      use_sentence_transformer (bool, optional): Flag to choose between BERT or SentenceTransformer. Defaults to False (BERT).

  Returns:
      float: The cosine similarity score between the two sentences (ranges from 0 to 1).
  """

  if use_sentence_transformer:
    # Use SentenceTransformer model for sentence embeddings
    sentence_embeddings = sentence_transformer_model.encode([sentence1, sentence2], convert_to_tensor=True)
  else:
    # Use BERT model for sentence embeddings
    sentence1_embedding = get_bert_embeddings(sentence1, use_sentence_transformer)
    sentence2_embedding = get_bert_embeddings(sentence2, use_sentence_transformer)
    sentence_embeddings = [sentence1_embedding, sentence2_embedding]

  # Calculate cosine similarity between the embeddings
  cosine_similarity = util.pytorch_cos_sim(sentence_embeddings[0], sentence_embeddings[1]).item()

  return cosine_similarity

#-------------------------------------------------------------------------------------------------------
def validate_revised_query(original_query, revised_query, threshold=0.6):
  """
  This function validates a revised query based on its similarity to the original query using SentenceTransformer embeddings and a cosine similarity threshold.

  Args:
      original_query (str): The original query.
      revised_query (str): The revised query.
      threshold (float, optional): The cosine similarity threshold for considering the revision successful. 
      Defaults to 0.6.

  Returns:
      bool: True if the revised query is considered valid (similar enough to the original query), False otherwise.
  """

  # Compute cosine similarity between original and revised queries using SentenceTransformer
  similarity = compute_bert_similarity(original_query, revised_query, use_sentence_transformer=True)
  print(f"Similarity: {similarity}")

  # Check if similarity is above the threshold (indicating a valid revision)
  return similarity > threshold