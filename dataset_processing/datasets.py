import pandas
from pandas import Series
from dataset_processing.daigt.process_dataset import DAIGTDataset
from dataset_processing.pratyushi.process_dataset import PratyushiDataset

TRAIN_DATASETS = [
    DAIGTDataset.get_randomly_sampled(
        human_sample_n=10000,
        ai_sample_n=10000
    )
]
EVAL_DATASETS = [PratyushiDataset.get()]


def get_train_dataset() -> tuple[Series, Series]:
    """Load training dataset and return features (strings) and labels (ints) respectively."""
    X, y = __concatenate_dataset(TRAIN_DATASETS)

    __print_dataset_composition(X, y, "Test")

    return X, y


def get_eval_dataset() -> tuple[Series, Series]:
    """Load evaluation dataset and return features (strings) and labels (ints) respectively."""
    X, y = __concatenate_dataset(EVAL_DATASETS)

    __print_dataset_composition(X, y, "Eval")

    return X, y


def __concatenate_dataset(
    datasets: list[tuple[Series, Series]],
) -> tuple[Series, Series]:
    X_list = []
    y_list = []

    for dataset in datasets:
        X_list.append(dataset[0])
        y_list.append(dataset[1])

    X = pandas.concat(X_list, axis=0)
    y = pandas.concat(y_list, axis=0).astype(int)

    return X, y

def __print_dataset_composition(X: Series, y: Series, dataset_name: str):
    print(f"{dataset_name} dataset composition: X.head(3) -> \n{X.head(3)}\n")
    print(f"{dataset_name} dataset composition: Y -> \n{y.value_counts()}\n")