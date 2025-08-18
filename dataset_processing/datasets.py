from numpy import ndarray
from dataset_processing.daigt.process_dataset import DAIGTDataset
from dataset_processing.pratyushi.process_dataset import PratyushiDataset

def get_train_dataset() -> tuple[ndarray, ndarray]:
    """Load training dataset and return features and labels respectively."""
    return DAIGTDataset.get()

def get_eval_dataset() -> tuple[ndarray, ndarray]:
    """Load evaluation dataset and return features and labels respectively."""
    return PratyushiDataset.get()