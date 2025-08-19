from pandas import read_parquet
from numpy import ndarray
from typing import Optional


class OkemdadDataset:

    @staticmethod
    def get(sample_n: Optional[int] = None) -> tuple[ndarray, ndarray]:
        """Load Okokemdad/ai_text_dataset dataset and return features and labels respectively.

        This dataset consists of 100% AI generated text.

        Dataset source: https://huggingface.co/datasets/okemdad/ai_text_dataset
        """
        dataset = read_parquet(
            "hf://datasets/okemdad/ai_text_dataset/data/train-00000-of-00001.parquet"
        )

        dataset.dropna(inplace=True)

        if sample_n:
            dataset = dataset.sample(sample_n)

        X = dataset["text"]
        y = dataset["label"]

        return X.to_numpy(), y.to_numpy()
