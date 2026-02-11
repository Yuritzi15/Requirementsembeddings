from transformers import BertTokenizer, BertModel
import torch  

class BertEmbedder:
    def __init__(self, model_name='bert-base-uncased'):
        self.tokenizer = BertTokenizer.from_pretrained(model_name)
        self.model = BertModel.from_pretrained(model_name)

    def embed(self, text):
        inputs = self.tokenizer(text, return_tensors='pt', truncation=True, padding=True)
        with torch.no_grad():
            outputs = self.model(**inputs)
            #mb_cls = outputs.last_hidden_state[:, 0, :]  # CLS token embedding

            attention_mask = inputs['attention_mask']
            hidden_states = outputs.last_hidden_state
            mask_expanded = attention_mask.unsqueeze(-1).expand(hidden_states.size()).float()
            sum_hidden = torch.sum(hidden_states * mask_expanded, 1)
            sum_mask = torch.clamp(mask_expanded.sum(1), min=1e-9)
            emb_mean = sum_hidden / sum_mask  # Mean pooling
        return emb_mean
        #eturn emb_cls, emb_mean