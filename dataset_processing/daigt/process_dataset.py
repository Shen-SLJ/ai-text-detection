from pandas import read_csv
from numpy import ndarray
from utils.path_utils import abs_path_from_project_path
from utils.dataset_utils import balance_two_class_dataset_by_dropping

DATASET_PATH = abs_path_from_project_path("dataset_processing/daigt/dataset.csv")


class DAIGTDataset:

    @staticmethod
    def get(balanced: bool = True) -> tuple[ndarray, ndarray]:
        """Load DAIGT dataset and return features and labels respectively.

        Dataset source: https://www.kaggle.com/datasets/thedrcat/daigt-v2-train-dataset
        """
        dataset = read_csv(DATASET_PATH)

        if balanced:
            dataset = balance_two_class_dataset_by_dropping(dataset, 'label')

        X = dataset["text"]
        y = dataset["label"]

        return X.to_numpy(), y.to_numpy()
