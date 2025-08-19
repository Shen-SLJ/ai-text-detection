from typing import List
from utils.path_utils import abs_path_from_project_path
from transformers import (
    RobertaForSequenceClassification,
    RobertaTokenizer,
    TrainingArguments,
    Trainer,
)
from datasets import Dataset
from torch import argmax
from torch.nn.functional import softmax
from utils.gpu_utils import is_gpu_available


class CandidateRobertaModel:
    """Text classification model using RoBERTa"""

    MODEL_SAVE_PATH = abs_path_from_project_path("saved/candidate_model_roberta")

    def __init__(
        self,
        train_documents: List[str],
        train_labels: List[int],
        load_from_saved: bool = False,
    ):
        self.train_documents = train_documents
        self.train_labels = train_labels
        self.tokenizer: RobertaTokenizer = RobertaTokenizer.from_pretrained(
            "roberta-base"
        )
        self.model = RobertaForSequenceClassification.from_pretrained(
            self.MODEL_SAVE_PATH if load_from_saved else "roberta-base"
        )

        self.use_cpu = not is_gpu_available()
        self.device = "cuda" if not self.use_cpu else "cpu"
        self.__model_use_cuda_or_warn()

    def __model_use_cuda_or_warn(self):
        if not self.use_cpu:
            self.model.to("cuda")

    def train(self) -> "CandidateRobertaModel":
        """
        Train model and save checkpoint.
        """
        training_args = TrainingArguments(
            output_dir=self.MODEL_SAVE_PATH, use_cpu=self.use_cpu
        )

        tokenized_train_docs = self.tokenizer(
            self.train_documents, padding=True, truncation=True, return_tensors="pt"
        )

        dataset_dict = {
            "input_ids": tokenized_train_docs["input_ids"],
            "attention_mask": tokenized_train_docs["attention_mask"],
            "labels": self.train_labels,
        }

        train_dataset = Dataset.from_dict(dataset_dict)

        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            processing_class=self.tokenizer,
        )

        trainer.train()

        return self

    def predict(self, documents: List[str]) -> List[int]:
        X = self.tokenizer(
            documents, padding=True, truncation=True, return_tensors="pt"
        )
        X = {k: v.to(self.device) for k, v in X.items()}

        output = self.model(**X)

        predictions = softmax(output.logits, dim=-1)
        predictions_list = argmax(predictions).tolist()

        return predictions_list
