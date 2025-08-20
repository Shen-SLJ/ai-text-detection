from pandas import read_parquet, Series
from numpy import ndarray
from typing import Optional


class OkemdadDataset:
    """Okemdad dataset. The dataset consists of 100% AI generated text

    Dataset source: https://huggingface.co/datasets/okemdad/ai_text_dataset
    """

    @staticmethod
    def get(sample_n: Optional[int] = None, random_state: Optional[int] = 0) -> tuple[Series, Series]:
        """Load Okokemdad/ai_text_dataset dataset and return features (strings) and labels (ints) respectively.

        This dataset consists of 100% AI generated text.
        """
        dataset = read_parquet(
            "hf://datasets/okemdad/ai_text_dataset/data/train-00000-of-00001.parquet"
        )

        dataset.dropna(inplace=True)

        if sample_n:
            dataset = dataset.sample(sample_n, random_state=random_state)

        X = dataset["text"]
        y = dataset["label"]

        return X, y
