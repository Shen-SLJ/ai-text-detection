from typing import List
from utils.pickle_utils import save_to_pickle
from utils.path_utils import abs_path_from_project_path
from numpy import ndarray
from transformers import RobertaForSequenceClassification, TrainingArguments, Trainer

class CandidateModel:
    """Candidate model for text classification using RoBERTa"""

    SAVED_MODEL_FILENAME = "candidate_model.pkl"

    def __init__(self, train_documents: List[str], train_labels: List[int]):
        self.train_documents = train_documents
        self.train_labels = train_labels

        self.model = RobertaForSequenceClassification.from_pretrained('roberta-base')

    def train(self) -> "CandidateModel":
        training_args = TrainingArguments(
            output_dir=abs_path_from_project_path("saved/candidate_model"),
        )
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=self.train_documents,
            eval_dataset=self.train_labels
        )
        
        return self

    def save(self) -> "CandidateModel":
        save_to_pickle(self, self.SAVED_MODEL_FILENAME)

        print(f"Candidate model saved to {self.SAVED_MODEL_FILENAME}")

        return self

    def predict(self, documents: List[str]) -> ndarray:

        return []
