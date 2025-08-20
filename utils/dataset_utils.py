from pandas import DataFrame
from typing import Optional, Union, Literal


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


def sample_dataset_based_on_column_value(
    dataset: DataFrame,
    sample_n: int,
    column_identifier: str,
    match_value: Union[int, str],
    match_type: Literal["match", "not_matching"],
    random_state: Optional[int] = 0,
) -> DataFrame:
    dataset_label_matched = (
        dataset[dataset[column_identifier] == match_value]
        if match_type == "match"
        else dataset[dataset[column_identifier] != match_value]
    )

    dataset_to_use = dataset_label_matched.sample(sample_n, random_state=random_state)

    return dataset_to_use
