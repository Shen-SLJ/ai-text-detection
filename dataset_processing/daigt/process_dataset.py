from pandas import read_csv
from numpy import ndarray
from utils.path_utils import abs_path_from_project_path

DATASET_PATH = abs_path_from_project_path("dataset_processing/daigt/dataset.csv")

class DAIGTDataset:

  @staticmethod
  def get() -> tuple[ndarray, ndarray]:
    """Load DAIGT dataset and return features and labels respectively.
    
    Dataset source: https://www.kaggle.com/datasets/thedrcat/daigt-v2-train-dataset
    """
    dataset = read_csv(DATASET_PATH)

    X = dataset["text"]
    y = dataset["label"]

    return X.to_numpy(), y.to_numpy() 
