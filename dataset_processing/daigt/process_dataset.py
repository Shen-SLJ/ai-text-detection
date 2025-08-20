import pandas
from pandas import read_csv, Series
from numpy import ndarray
from utils.path_utils import abs_path_from_project_path
from utils.dataset_utils import (
    sample_dataset_based_on_column_value,
    correct_imbalance_by_dropping,
)
from typing import Optional, Literal

DATASET_PATH = abs_path_from_project_path("dataset_processing/daigt/dataset.csv")


class DAIGTDataset:
    """DAIGT dataset.

    Source: https://www.kaggle.com/datasets/thedrcat/daigt-v2-train-dataset
    """

    @staticmethod
    def get(balanced: bool = False) -> tuple[ndarray, ndarray]:
        """Load dataset and return features and labels respectively."""

        dataset = read_csv(DATASET_PATH)

        if balanced:
            dataset = correct_imbalance_by_dropping(dataset, "label")

        X = dataset["text"]
        y = dataset["label"]

        return X.to_numpy(), y.to_numpy()

    @staticmethod
    def get_randomly_sampled(
        human_sample_n: int = None,
        human_random_state: Optional[int] = 0,
        ai_sample_n: int = None,
        ai_random_state: Optional[int] = 0,
    ) -> tuple[Series, Series]:
        """Load dataset and get randomly sampled entries across the dataset for both ai generated & human generated partitions. Returns
        feature input and labels respectively.
        """
        dataset = read_csv(DATASET_PATH)

        human_dataset = sample_dataset_based_on_column_value(
            dataset=dataset,
            sample_n=human_sample_n,
            column_identifier="label",
            match_value=0,
            match_type="match",
            random_state=human_random_state,
        )

        ai_dataset = human_dataset = sample_dataset_based_on_column_value(
            dataset=dataset,
            sample_n=ai_sample_n,
            column_identifier="label",
            match_value=1,
            match_type="match",
            random_state=ai_random_state,
        )

        combined_dataset = pandas.concat(
            [human_dataset, ai_dataset], axis=0, ignore_index=True
        )

        return combined_dataset["text"], combined_dataset["label"]
