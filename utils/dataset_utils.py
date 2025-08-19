from pandas import DataFrame


def correct_imbalance_by_dropping_and_save(dataset: DataFrame, label_identifier: str, save_path: str) -> None:
    label_counts = dataset[label_identifier].value_counts()
    minority_label_count = label_counts[label_counts.idxmin()]
    majority_label = label_counts.idxmax()
    
    dataset_majority = dataset[dataset[label_identifier] == majority_label]

    num_rows_to_drop = len(dataset_majority) - minority_label_count
    rows_to_drop = dataset_majority.sample(num_rows_to_drop).index

    balanced_dataset = dataset.drop(rows_to_drop)

    balanced_dataset.to_csv(save_path)