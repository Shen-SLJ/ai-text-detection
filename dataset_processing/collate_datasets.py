from numpy import ndarray
from dataset_processing.daigt.process_dataset import DAIGTDataset

def get_collated_dataset() -> tuple[ndarray, ndarray]:
    """Load collated dataset and return features and labels respectively."""
    return DAIGTDataset.get()