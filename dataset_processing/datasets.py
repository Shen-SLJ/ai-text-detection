import pandas
from pandas import Series
from dataset_processing.daigt.process_dataset import DAIGTDataset
from dataset_processing.pratyushi.process_dataset import PratyushiDataset
from dataset_processing.okemdad_ai.process_dataset import OkemdadDataset

TRAIN_DATASETS = [
    DAIGTDataset.get(),
    OkemdadDataset.get(sample_n=9000)
]
EVAL_DATASETS = [PratyushiDataset.get()]


def get_train_dataset() -> tuple[Series, Series]:
    """Load training dataset and return features (strings) and labels (ints) respectively."""
    X, y = __concatenate_dataset(TRAIN_DATASETS)

    print(f"Train dataset composition: Y -> {y.value_counts()}")

    return X, y


def get_eval_dataset() -> tuple[Series, Series]:
    """Load evaluation dataset and return features (strings) and labels (ints) respectively."""
    X, y = __concatenate_dataset(EVAL_DATASETS)

    print(f"Eval dataset composition: Y -> {y.value_counts()}")

    return X, y


def __concatenate_dataset(
    datasets: list[tuple[Series, Series]],
) -> tuple[Series, Series]:
    X_list = []
    y_list = []

    for dataset in datasets:
        X_list.append(dataset[0])
        y_list.append(dataset[1])

    X = pandas.concat(X_list, axis=0, ignore_index=True)
    y = pandas.concat(y_list, axis=0, ignore_index=True)

    return X, y
