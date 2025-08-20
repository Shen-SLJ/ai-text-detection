from pandas import read_csv
from numpy import ndarray
from utils.path_utils import abs_path_from_project_path
from utils.dataset_utils import (
    get_dataset_or_sample_dataset_with_label,
    correct_imbalance_by_dropping,
)
from typing import Optional, Literal

DATASET_PATH = abs_path_from_project_path("dataset_processing/daigt/dataset.csv")


class DAIGTDataset:

    @staticmethod
    def get(
        balanced: bool = False,
        sample_n: Optional[int] = None,
        with_label: Literal["0", "1", None] = None,
        random_state: Optional[int] = 0,
    ) -> tuple[ndarray, ndarray]:
        """Load DAIGT dataset and return features and labels respectively.

        Dataset source: https://www.kaggle.com/datasets/thedrcat/daigt-v2-train-dataset

        Args:
            balanced: Whether to balance dataset. Will occur before sampling.
        """

        dataset = read_csv(DATASET_PATH)

        if balanced:
            dataset = correct_imbalance_by_dropping(dataset, "label")

        dataset = get_dataset_or_sample_dataset_with_label(
            dataset=dataset,
            sample_n=sample_n,
            with_label=with_label,
            random_state=random_state,
        )

        X = dataset["text"]
        y = dataset["label"]

        return X.to_numpy(), y.to_numpy()
