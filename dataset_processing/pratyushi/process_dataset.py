from pandas import read_csv
from numpy import ndarray
from utils.path_utils import abs_path_from_project_path

DATASET_PATH = abs_path_from_project_path("dataset_processing/pratyushi/dataset.csv")

class PratyushiDataset:

  @staticmethod
  def get() -> tuple[ndarray, ndarray]:
    """Load Pratyishi dataset and return features and labels respectively.
    
    Dataset source: https://www.kaggle.com/datasets/pratyushpuri/ai-vs-human-content-detection-1000-record-in-2025
    """
    dataset = read_csv(DATASET_PATH)

    X = dataset["text_content"]
    y = dataset["label"]

    return X.to_numpy(), y.to_numpy() 
