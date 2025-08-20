import pandas
from pandas import Series, read_parquet
from utils.path_utils import abs_path_from_project_path
from utils.dataset_utils import sample_dataset_based_on_column_value
from typing import Optional


DATASET_PATH = abs_path_from_project_path("dataset_processing/daigt/dataset.parquet")


class HumanVsLLMDataset:
    """
    Human vs LLM Dataset.

    Source: https://www.kaggle.com/datasets/starblasters8/human-vs-llm-text-corpus
    """

    @staticmethod
    def get_randomly_sampled(
        human_sample_n: int = None,
        human_random_state: Optional[int] = 0,
        ai_sample_n: int = None,
        ai_random_state: Optional[int] = 0,
    ) -> tuple[Series, Series]:
        """Load dataset and get randomly sampled entries across the dataset for both ai generated & human generated partitions. Returns
        feature input (strings) and labels (ints) respectively.
        """
        dataset = read_parquet(DATASET_PATH)

        human_dataset = sample_dataset_based_on_column_value(
            dataset=dataset,
            sample_n=human_sample_n,
            column_identifier="source",
            match_value="Human",
            match_type="match",
            random_state=human_random_state,
        )

        ai_dataset = sample_dataset_based_on_column_value(
            dataset=dataset,
            sample_n=ai_sample_n,
            column_identifier="source",
            match_value="Human",
            match_type="not_matching",
            random_state=ai_random_state,
        )

        combined_dataset = pandas.concat(
            [human_dataset, ai_dataset], axis=0
        )

        combined_dataset.loc[combined_dataset["source"] == "Human", "source"] = 0
        combined_dataset.loc[combined_dataset["source"] != "Human", "source"] = 1

        return combined_dataset["text"], combined_dataset["source"]
