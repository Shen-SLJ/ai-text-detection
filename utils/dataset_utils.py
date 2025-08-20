from pandas import DataFrame
from typing import Optional, Literal


def correct_imbalance_by_dropping(
    dataset: DataFrame, label_identifier: str, random_state: Optional[int] = 0
) -> None:
    """Corrects dataset imbalance by dropping rows of the majority class until entries are equal.
    Random state is default set to 0 for reproducability,
    """
    label_counts = dataset[label_identifier].value_counts()
    minority_label_count = label_counts[label_counts.idxmin()]
    majority_label = label_counts.idxmax()

    dataset_majority = dataset[dataset[label_identifier] == majority_label]

    num_rows_to_drop = len(dataset_majority) - minority_label_count
    rows_to_drop = dataset_majority.sample(
        num_rows_to_drop, random_state=random_state
    ).index

    return dataset.drop(rows_to_drop)


def get_dataset_or_sample_dataset_with_label(
    dataset: DataFrame,
    sample_n: Optional[int] = None,
    with_label: Optional[str] = None,
    random_state: Optional[int] = 0,
) -> DataFrame:
    """
    Sample dataset randomly.

    Args:
        sample_n: Sample n number of entries from dataset with the label = with_label.
        with_label: Set to get entries with the label only. Don't set to consider all labels.
    """
    dataset_to_use = dataset
    dataset_matching_label = dataset[dataset["label"] == with_label]

    if sample_n:
        if with_label:
            dataset_to_use = dataset_matching_label.sample(
                sample_n, random_state=random_state
            )
        else:
            dataset_to_use = dataset.sample(sample_n, random_state=random_state)
    elif with_label:
        dataset_to_use = dataset_matching_label

    return dataset_to_use
