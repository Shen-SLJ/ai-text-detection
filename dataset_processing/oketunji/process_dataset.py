# https://huggingface.co/datasets/dmitva/human_ai_generated_text

from typing import Optional
from pandas import Series, read_parquet, DataFrame
from utils.path_utils import abs_path_from_project_path
import pandas as pd

DATASET_PATH = abs_path_from_project_path("dataset_processing/oketunji/part0.parquet")


class OketunjiDataset:
    """
    Dataset from Evaluating the Efficacy of Hybrid Deep Learning Models in Distinguishing AI-Generated Text (2023)
    by Abiodun Finbarrs Oketunji.

    Contains academic essays written by humans and AI (ChatGPT4, ChatGPT3.5, LLaMA2, PaLM2) on the same topics.

    Source: https://huggingface.co/datasets/dmitva/human_ai_generated_text
    """

    __X_NAME = "text"
    __Y_NAME = "label"

    @staticmethod
    def get_randomly_sampled(
        human_sample_n: int,
        ai_sample_n: int,
        human_random_state: Optional[int] = 0,
        ai_random_state: Optional[int] = 0,
    ) -> tuple[Series, Series]:
        dataset = read_parquet(DATASET_PATH)

        if ai_sample_n > len(dataset) or human_sample_n > len(dataset):
            raise ValueError(
                f"Sample size cannot be greater than dataset size. (max: {len(dataset)})"
            )

        human_series = dataset["human_text"].sample(
            n=human_sample_n, random_state=human_random_state
        )
        ai_series = dataset["ai_text"].sample(
            n=ai_sample_n, random_state=ai_random_state
        )

        human_df = DataFrame(human_series)
        human_df.rename(columns={"human_text": OketunjiDataset.__X_NAME}, inplace=True)
        human_df["label"] = 0

        ai_df = DataFrame(ai_series)
        ai_df.rename(columns={"ai_text": OketunjiDataset.__X_NAME}, inplace=True)
        ai_df["label"] = 1

        combined_df = pd.concat([human_df, ai_df], axis=0).sample(
            frac=1, random_state=0
        ).reset_index(drop=True)

        return (
            combined_df[OketunjiDataset.__X_NAME],
            combined_df[OketunjiDataset.__Y_NAME],
        )
