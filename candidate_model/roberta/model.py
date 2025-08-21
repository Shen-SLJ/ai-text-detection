from typing import List, Optional, Literal, Union
from utils.path_utils import abs_path_from_project_path
from transformers import (
    RobertaForSequenceClassification,
    RobertaTokenizer,
    RobertaConfig,
    TrainingArguments,
    Trainer,
    DataCollatorWithPadding,
)
from transformers.trainer_utils import EvalPrediction
from datasets import Dataset
from torch.utils.data import DataLoader
from utils.gpu_utils import is_gpu_available
from utils.metric_utils import get_false_positive_rate
from sklearn.metrics import confusion_matrix
import evaluate
import numpy as np
import torch


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

        self.accuracy_metric = evaluate.load("accuracy")
        self.recall_metric = evaluate.load("recall")

        print(
            f"len_train: {len(self.train_documents)}, len_eval: {len(self.eval_documents)}"
        )

        self.tokenizer: RobertaTokenizer = RobertaTokenizer.from_pretrained(
            "roberta-base"
        )
        self.model = RobertaForSequenceClassification.from_pretrained(
            pretrained_model_name_or_path=self.__get_pretrained_model_name_or_path(
                load_checkpoint_number
            ),
            config=self.__get_model_config(),
        )

        self.use_cpu = not is_gpu_available()
        self.device = "cuda" if not self.use_cpu else "cpu"
        self.__model_use_cuda_or_warn()

    def __model_use_cuda_or_warn(self):
        if not self.use_cpu:
            self.model.to("cuda")

    def __get_pretrained_model_name_or_path(
        self,
        load_checkpoint_number: Optional[int],
    ) -> str:
        return (
            f"{self.MODEL_SAVE_PATH}/checkpoint-{load_checkpoint_number}"
            if load_checkpoint_number
            else "roberta-base"
        )

    def __get_model_config(self) -> RobertaConfig:
        config = RobertaConfig.from_pretrained("roberta-base")
        config.hidden_dropout_prob = 0.2
        config.attention_probs_dropout_prob = 0.2

        return config

    def train(self) -> "CandidateRobertaModel":
        """
        Train model and save checkpoint.
        """
        training_args = TrainingArguments(
            output_dir=self.MODEL_SAVE_PATH,
            save_total_limit=5,
            use_cpu=self.use_cpu,
            eval_strategy="steps",
            eval_steps=100,
            logging_steps=100,
            save_steps=250,
            num_train_epochs=3,
            per_device_train_batch_size=8,
            per_device_eval_batch_size=8,
            gradient_accumulation_steps=8,
            learning_rate=3e-5,
            weight_decay=0.05,
        )

        train_dataset = self.__get_dataset(self.train_documents, self.train_labels)
        eval_dataset = self.__get_dataset(self.eval_documents, self.eval_labels)

        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            processing_class=self.tokenizer,
            compute_metrics=self.__compute_training_metrics,
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

    def __compute_training_metrics(
        self, eval_preds: EvalPrediction
    ) -> dict[str, float]:
        logits, labels = eval_preds
        predictions = self.__get_predictions_from_logits(logits, logits_type="np")

        accuracy = self.accuracy_metric.compute(
            predictions=predictions, references=labels
        )
        recall = self.recall_metric.compute(predictions=predictions, references=labels)

        tn, fp, _, _ = confusion_matrix(labels, predictions).ravel()
        fpr = get_false_positive_rate(fp, tn)

        return {
            "accuracy": accuracy["accuracy"],
            "recall": recall["recall"],
            "fpr": fpr,
        }

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

        self.model.eval()

        all_predictions = []

        for batch in X_data_loader:
            batch = {k: v.to(self.device) for k, v in batch.items()}

            with torch.no_grad():
                output = self.model(**batch)

            predictions_list = self.__get_predictions_from_logits(
                output.logits, logits_type="pt"
            )
            all_predictions.extend(predictions_list)

        return all_predictions

    def __get_predictions_from_logits(
        self, logits: Union[torch.Tensor, np.ndarray], logits_type: Literal["pt", "np"]
    ) -> List[int]:
        if logits_type == "pt":
            predictions_list = torch.argmax(logits, dim=-1).tolist()
        else:
            predictions_list = np.argmax(logits, axis=-1)

        return predictions_list
