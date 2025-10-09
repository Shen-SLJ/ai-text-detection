import nltk
import numpy as np
from textatistic import Textatistic
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


def get_flesch_reading_ease_score(
    document: str, should_log_exceptions: bool = False
) -> float:
    """Calculate the Flesch Reading Ease score for a given document."""
    try:
        text_metrics = Textatistic(text=document)
        return text_metrics.flesch_score
    except Exception as e:
        if should_log_exceptions:
            print(
                f"Error calculating Flesch Reading Ease score: {e}. Document={document}"
            )
        return 0.0


def get_document_metrics_as_feature(
    documents: list[str], metric_func: Callable[[str], int]
) -> np.ndarray:
    """Returns (N, 1) shaped array of the metric for each document."""
    metrics = []

    for i, doc in enumerate(documents):
        print(
            f"Processing document {i+1}/{len(documents)}, metric_func={metric_func.__name__}",
            end="\r",
        )
        metrics.append(metric_func(doc))

    print("")

    return np.array(metrics).reshape(-1, 1)
