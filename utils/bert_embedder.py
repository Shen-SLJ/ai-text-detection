import numpy
import torch
from transformers import BertTokenizer, BertModel
from typing import Union


class BERTEmbedder:
    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"BERT embedder using device: {self.device}")

        self.tokenizer = BertTokenizer.from_pretrained(
            "bert-base-uncased"
        )  # https://huggingface.co/docs/transformers/v4.51.3/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.from_pretrained
        self.model = BertModel.from_pretrained("bert-base-uncased").to(self.device)

    def get_bert_embedding(self, text: Union[str, list[str]]) -> numpy.ndarray:
        """
        Get BERT embeddings of input text. Embedding obtained from BERT [CLS] token embedding.
        """
        input = self.tokenizer(
            text=text, padding=True, truncation=True, return_tensors="pt"
        ).to(
            self.device
        )  # encode_plus(), return torch.Tensor

        with torch.no_grad():  # speed increase
            outputs = self.model(
                input_ids=input["input_ids"],
                token_type_ids=input["token_type_ids"],
                attention_mask=input["attention_mask"],
            )  # BertModel.forward(), https://huggingface.co/docs/transformers/en/model_doc/bert#transformers.BertModel.forward

            # Use [CLS] token embedding, which contains context of entire sentence. .cpu() moves tensor from GPU to CPU mem
            embeddings = outputs.last_hidden_state[:, 0, :].cpu().numpy()  

            return embeddings
