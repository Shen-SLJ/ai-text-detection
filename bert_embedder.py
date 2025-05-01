import torch
from transformers import BertTokenizer, BertModel

tokenizer = BertTokenizer.from_pretrained(
    "bert-base-uncased"
)  # https://huggingface.co/docs/transformers/v4.51.3/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.from_pretrained
model = BertModel.from_pretrained("bert-base-uncased")

# Encode some text
text = "Test sentence for BERT embedding, and this is an amazing second sentence."
input = tokenizer(text=text, return_tensors="pt")  # encode_plus(), return torch.Tensor

with torch.no_grad():  # speed increase
    outputs = model(
        input_ids=input["input_ids"],
        token_type_ids=input["token_type_ids"],
        attention_mask=input["attention_mask"],
    )  # BertModel.forward(), https://huggingface.co/docs/transformers/en/model_doc/bert#transformers.BertModel.forward
    embeddings = outputs.last_hidden_state[0, :]
