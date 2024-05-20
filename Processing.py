from transformers import BertTokenizer, BertModel
import torch


class TextEmbedder:
    def __init__(self):
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        self.model = BertModel.from_pretrained('bert-base-uncased')

    def _mean_pooling(self, model_output, attention_mask):
        token_embeddings = model_output.last_hidden_state
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)
        sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
        return sum_embeddings / sum_mask

    def embed_text(self, examples):
        inputs = self.tokenizer(
            examples["content"], padding=True, truncation=True, return_tensors="pt"
        )
        with torch.no_grad():
            model_output = self.model(**inputs)
        pooled_embeds = self._mean_pooling(model_output, inputs["attention_mask"])
        return {"embedding": pooled_embeds.cpu().numpy()}
    
    def generate_embeddings(self, dataset):
        return dataset.map(self.embed_text, batched=True, batch_size=128)
    
    def embed_query(self, query_text):
        query_inputs = self.tokenizer(
            query_text,
            padding=True,
            truncation=True,
            return_tensors="pt"
        )

        with torch.no_grad():
            query_model_output = self.model(**query_inputs)

        query_embedding = self._mean_pooling(query_model_output, query_inputs["attention_mask"])

        return query_embedding