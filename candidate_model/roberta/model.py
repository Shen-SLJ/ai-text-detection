from typing import List, Optional
from utils.path_utils import abs_path_from_project_path
from transformers import (
    RobertaForSequenceClassification,
    RobertaTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorWithPadding,
)
from datasets import Dataset
from torch import argmax
from torch.nn.functional import softmax
from torch.utils.data import DataLoader
from utils.gpu_utils import is_gpu_available


class CandidateRobertaModel:
    """Text classification model using RoBERTa

    Args:
        load_checkpoint_number: If specified will load the model with the specific checkpoint number
    """

    MODEL_SAVE_PATH = abs_path_from_project_path("saved/candidate_model_roberta")
    PREDICTION_BATCH_SIZE = 8

    def __init__(
        self,
        train_documents: List[str] = [],
        train_labels: List[int] = [],
        eval_documents: List[str] = [],
        eval_labels: List[int] = [],
        load_checkpoint_number: Optional[int] = None,
    ):
        self.train_documents = train_documents
        self.train_labels = train_labels
        self.eval_documents = eval_documents
        self.eval_labels = eval_labels

        self.tokenizer: RobertaTokenizer = RobertaTokenizer.from_pretrained(
            "roberta-base"
        )
        self.model = RobertaForSequenceClassification.from_pretrained(
            f"{self.MODEL_SAVE_PATH}/checkpoint-{load_checkpoint_number}"
            if load_checkpoint_number
            else "roberta-base"
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
            output_dir=self.MODEL_SAVE_PATH, 
            use_cpu=self.use_cpu,
            eval_strategy="steps",
            eval_steps=100,
            num_train_epochs=3,
            per_device_train_batch_size=16,
            per_device_eval_batch_size=16,
            learning_rate=5e-6
        )

        train_dataset = self.__get_dataset(self.train_documents, self.train_labels)
        eval_dataset = self.__get_dataset(self.eval_documents, self.eval_labels)

        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            processing_class=self.tokenizer,
        )

        trainer.train()

        return self

    def __get_dataset(self, docs: List[str], labels: List[int]) -> Dataset:
        tokenized_docs = self.tokenizer(
            docs, padding=True, truncation=True, return_tensors="pt"
        )
        dataset_dict = {
            "input_ids": tokenized_docs["input_ids"],
            "attention_mask": tokenized_docs["attention_mask"],
            "labels": labels,
        }

        dataset = Dataset.from_dict(dataset_dict)

        return dataset

    def predict(self, documents: List[str]) -> List[int]:
        """Run prediction. Documents automatically batched."""
        X = self.tokenizer(
            documents, padding=True, truncation=True, return_tensors="pt"
        )

        collator = DataCollatorWithPadding(self.tokenizer)
        X_dataset = Dataset.from_dict(X)
        X_data_loader = DataLoader(
            dataset=X_dataset,
            batch_size=self.PREDICTION_BATCH_SIZE,
            collate_fn=collator,
        )

        all_predictions = []

        for batch in X_data_loader:
            batch = {k: v.to(self.device) for k, v in X.items()}
            
            output = self.model(**batch)

            predictions = softmax(output.logits, dim=-1)
            predictions_list = argmax(predictions, dim=-1).tolist()

            all_predictions.extend(predictions_list)

        return all_predictions
