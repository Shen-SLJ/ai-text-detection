import numpy as np
from numpy import ndarray
from dataset_processing.daigt.process_dataset import DAIGTDataset
from dataset_processing.pratyushi.process_dataset import PratyushiDataset
from dataset_processing.okemdad_ai.process_dataset import OkemdadDataset

TRAIN_DATASETS = [
    DAIGTDataset.get()
]
EVAL_DATASETS = [
    PratyushiDataset.get()
]

def get_train_dataset() -> tuple[ndarray, ndarray]:
    """Load training dataset and return features and labels respectively."""
    return DAIGTDataset.get()

def get_eval_dataset() -> tuple[ndarray, ndarray]:
    """Load evaluation dataset and return features and labels respectively."""
    X, y = __concatenate_dataset(EVAL_DATASETS)

    return X, y

# TODO: This breaks training for transformers for some reason
def __concatenate_dataset(dataset: list[tuple[ndarray, ndarray]]) -> tuple[ndarray, ndarray]:
    X = np.array([])
    y = np.array([])

    for dataset in dataset:
        dataset_X = dataset[0]
        dataset_y = dataset[1]

        X = np.concatenate((X, dataset_X))
        y = np.concatenate((y, dataset_y))

    return X, y
    