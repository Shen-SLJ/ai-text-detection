from __future__ import annotations

import numpy as np
import pandas as pd
from utils.path_utils import abs_path_from_project_path

class DatasetPreprocessor:
    __DATASET_PATHS: list[str] = [
        "datasets/dataset-essays-1.csv"
    ]

    def __init__(self):
        self.__combined_dataset: np.ndarray = np.empty(shape=[0,0])

    def load_datasets(self) -> DatasetPreprocessor:
        # TODO: Convert into a list reading function, and implement cleaning later
        dataset = pd.read_csv(abs_path_from_project_path(self.__DATASET_PATHS[0])) # Temp index 0
        dataset_numpy = dataset.to_numpy()

        self.__combined_dataset = dataset_numpy

        return self


    def combined_dataset_as_numpy_array(self):
        return self.__combined_dataset