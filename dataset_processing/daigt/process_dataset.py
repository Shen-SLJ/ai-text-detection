from pandas import read_csv
from numpy import ndarray
from utils.path_utils import abs_path_from_project_path
from utils.dataset_utils import correct_imbalance_by_dropping
from typing import Optional, Literal

DATASET_PATH = abs_path_from_project_path("dataset_processing/daigt/dataset.csv")
DATASET_BALANCED_PATH = abs_path_from_project_path(
    "dataset_processing/daigt/dataset-balanced.csv"
)


class DAIGTDataset:

    @staticmethod
    def get(
        balanced: bool = False,
        sample_n: Optional[int] = None,
        from_label: Literal["0", "1", None] = None,
        random_state: Optional[int] = 0,
    ) -> tuple[ndarray, ndarray]:
        """Load DAIGT dataset and return features and labels respectively.

        Dataset source: https://www.kaggle.com/datasets/thedrcat/daigt-v2-train-dataset

        Args:
            balanced: Set to balance the dataset. Will occur before any sampling.
            sample_n: Sample n number of entries from the dataset. If from_label is set, will sample n entries from that label.
            from_label: Set to get entries with the label only. 
        """
        dataset_to_use = None

        dataset = read_csv(DATASET_BALANCED_PATH if balanced else DATASET_PATH)
        dataset_matching_label = dataset[dataset["label"] == from_label]

        if balanced:
            dataset = correct_imbalance_by_dropping(dataset, "label")

        if sample_n:
            if from_label:
                dataset_to_use = dataset_matching_label.sample(
                    sample_n, random_state=random_state
                )
            else:
                dataset_to_use = dataset.sample(sample_n, random_state=random_state)
        elif from_label:
            dataset_to_use = dataset_matching_label

        X = dataset_to_use["text"]
        y = dataset_to_use["label"]

        return X.to_numpy(), y.to_numpy()
