from data_preprocessor import DataPreprocessor

if __name__ == '__main__':
    data_preprocessor = DataPreprocessor()
    dataset_numpy = data_preprocessor.load_datasets().combined_dataset_as_numpy_array()

    features = dataset_numpy[:, 0]
    labels = dataset_numpy[:, 1]