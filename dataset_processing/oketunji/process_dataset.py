# https://huggingface.co/datasets/dmitva/human_ai_generated_text

from typing import Optional
from pandas import Series, read_parquet
from utils.path_utils import abs_path_from_project_path

DATASET_PATH = abs_path_from_project_path("dataset_processing/oketunji/part0.parquet")


class OketunjiDataset:
    """
    Dataset from Evaluating the Efficacy of Hybrid Deep Learning Models in Distinguishing AI-Generated Text (2023)
    by Abiodun Finbarrs Oketunji.

    Contains academic essays written by humans and AI (ChatGPT4, ChatGPT3.5, LLaMA2, PaLM2) on the same topics.

    Source: https://huggingface.co/datasets/dmitva/human_ai_generated_text
    """

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

        return human_series, ai_series
