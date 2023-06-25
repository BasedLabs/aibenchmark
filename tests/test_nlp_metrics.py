import pytest
from aibenchmark.metrics import NLPMetrics


def test_compute_perplexity():
    predicted_logits = [[0.87, 0.52, 0.7, 0.12, 0.89], [0.88, 0.74, 0.51, 0.9, 0.23]]
    target_ids = [[1, 0, 1, 0, 1], [1, 1, 0, 1, 0]]
    nlp_metrics = NLPMetrics()
    perplexity = nlp_metrics.compute_perplexity(predicted_logits, target_ids)
    assert (perplexity == 112.75840541227849)
    assert isinstance(perplexity, float)
    assert perplexity > 0