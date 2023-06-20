import pytest
from aibenchmark.metrics import NLPMetrics

@pytest.fixture
def nlp_metrics():
    references = [
        "The cat is on the mat",
        "There is a cat on the mat"
    ]
    return NLPMetrics(references)

def test_compute_perplexity(nlp_metrics):
    text = "The cat is on the mat"
    perplexity = nlp_metrics.compute_perplexity(text)
    assert (perplexity == 3.965406456500188)
    assert isinstance(perplexity, float)
    assert perplexity > 0

def test_compute_bleu(nlp_metrics):
    text = "The cat is on the mat"
    bleu_score = nlp_metrics.compute_bleu(text)
    assert isinstance(bleu_score, float)
    assert bleu_score >= 0 and bleu_score <= 1

def test_compute_rouge(nlp_metrics):
    text = "The cat is on the mat"
    rouge_score = nlp_metrics.compute_rouge(text)
    assert isinstance(rouge_score, float)
    assert rouge_score >= 0 and rouge_score <= 1