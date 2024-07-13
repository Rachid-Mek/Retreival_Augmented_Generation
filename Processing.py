from transformers import BertTokenizer, BertModel
import torch

class TextEmbedder:
  """
  This class embeds text using a pre-trained BERT model.
  """

  def __init__(self):
    """
    Initializes the TextEmbedder object with a pre-trained BERT model and tokenizer.
    """
    self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    self.model = BertModel.from_pretrained('bert-base-uncased')


  def _mean_pooling(self, model_output, attention_mask):
    """
    Performs mean pooling on the last hidden state of the BERT model output, weighted by the attention mask.

    Args:
        model_output: The output dictionary from the BERT model.
        attention_mask: The attention mask for the input sequence.

    Returns:
        torch.Tensor: The sentence embedding as a PyTorch tensor.
    """

    token_embeddings = model_output.last_hidden_state  
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float() 

    # Weight token embeddings by attention mask and calculate sum
    sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)
    sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)  

    # Normalize the sum by the number of valid tokens (weighted by attention mask)
    return sum_embeddings / sum_mask


  def embed_text(self, examples):
    """
    Embeds a list of text examples using the BERT model.

    Args:
        examples (list): A list of dictionaries where each dictionary has a "content" key with the text content to embed.

    Returns:
        dict: A dictionary with an "embedding" key containing a NumPy array of the sentence embeddings.
    """

    inputs = self.tokenizer( 
        examples["content"], padding=True, truncation=True, return_tensors="pt"
    )

    with torch.no_grad(): 
      model_output = self.model(**inputs)
      pooled_embeds = self._mean_pooling(model_output, inputs["attention_mask"])  
    return {"embedding": pooled_embeds.cpu().numpy()}  

  def generate_embeddings(self, dataset):
    """
    Generates embeddings for a dataset using batched processing for efficiency.

    Args:
        dataset (transformers.data.Dataset): A Hugging Face Dataset object.

    Returns:
        transformers.data.Dataset: The same dataset object with an additional "embedding" column containing the sentence embeddings.
    """

    return dataset.map(self.embed_text, batched=True, batch_size=128)  

  def embed_query(self, query_text):
    """
    Embeds a single query text using the BERT model.

    Args:
        query_text (str): The query text to embed.

    Returns:
        torch.Tensor: The query embedding as a PyTorch tensor.
    """

    query_inputs = self.tokenizer( 
        query_text, padding=True, truncation=True, return_tensors="pt"
    )

    with torch.no_grad():  
      query_model_output = self.model(**query_inputs)  
    query_embedding = self._mean_pooling(query_model_output, query_inputs["attention_mask"])  

    return query_embedding