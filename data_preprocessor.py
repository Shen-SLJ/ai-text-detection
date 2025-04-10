import pandas as pd
from utils.PathUtils import abs_path_from_project_path

class DataPreprocessor:
    __DATASET_PATHS: list[str] = [
        "datasets/dataset-essays-1.csv"
    ]

    def __init__(self):
        self.__combined_dataset = None

        self.__load_datasets()

    def __load_datasets(self) -> None:
        dataset = pd.read_csv(abs_path_from_project_path(self.__DATASET_PATHS[0])) # Temp index 0
        print(dataset['generated'][0])