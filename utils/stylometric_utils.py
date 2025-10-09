import nltk
import numpy as np
from readability import Readability
from typing import Callable

nltk.download("punkt_tab")


def get_sentence_burstiness_score(document: str) -> float:
    sentences = nltk.tokenize.sent_tokenize(document)

    if len(sentences) == 0:
        return 0.0

    sentence_lengths = [len(sentence.split()) for sentence in sentences]
    std = np.std(sentence_lengths)
    normalized_std = (
        std / np.mean(sentence_lengths) if np.mean(sentence_lengths) > 0 else 0.0
    )

    return normalized_std


def get_flesch_reading_ease_score(document: str) -> float:
    """Calculate the Flesch Reading Ease score for a given document.
    """
    r = Readability(document)

    return r.flesch_reading_ease()

def get_document_metrics_as_feature(document: str, metric_func: Callable[[str], int]) -> np.ndarray:
    """Returns (N, 1) np array of the metric for each document."""
    return np.array([metric_func(document)]).reshape(-1, 1)