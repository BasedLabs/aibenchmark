import pytest
import torch
import pandas as pd
from aibenchmark.metrics import ClassificationMetrics

@pytest.fixture
def binary_data():
    predictions = [1, 0, 1, 1, 0]
    targets = [1, 1, 1, 0, 0]
    yield predictions, targets

@pytest.fixture
def multiclass_data():
    predictions = [0, 2, 1, 1, 0, 2]
    targets = [0, 1, 1, 2, 0, 2]
    yield predictions, targets

def test_metrics_accuracy_binary(binary_data):
    predictions, targets = binary_data
    metrics = ClassificationMetrics(predictions, targets, average="binary")
    accuracy = metrics.calculate_accuracy()
    assert accuracy == pytest.approx(0.6)

def test_metrics_accuracy_multiclass(multiclass_data):
    predictions, targets = multiclass_data
    metrics = ClassificationMetrics(predictions, targets, average="macro")
    accuracy = metrics.calculate_accuracy()
    assert accuracy == pytest.approx(0.6666666666666666)

def test_metrics_precision_binary(binary_data):
    predictions, targets = binary_data
    metrics = ClassificationMetrics(predictions, targets, average="binary")
    precision = metrics.calculate_precision()
    assert precision == pytest.approx(0.6666666)

def test_metrics_precision_multiclass(multiclass_data):
    predictions, targets = multiclass_data
    metrics = ClassificationMetrics(predictions, targets, average="micro")
    precision = metrics.calculate_precision()
    assert precision == pytest.approx(0.6666666666666666)

def test_metrics_recall_binary(binary_data):
    predictions, targets = binary_data
    metrics = ClassificationMetrics(predictions, targets, average="binary")
    recall = metrics.calculate_recall()
    assert recall == pytest.approx(0.6666666666666666)

def test_metrics_recall_multiclass(multiclass_data):
    predictions, targets = multiclass_data
    metrics = ClassificationMetrics(predictions, targets, average="micro")
    recall = metrics.calculate_recall()
    assert recall == pytest.approx(0.6666666666666666)

def test_metrics_f1_score_binary(binary_data):
    predictions, targets = binary_data
    metrics = ClassificationMetrics(predictions, targets, average="binary")
    f1_score = metrics.calculate_f1_score()
    assert f1_score == pytest.approx(0.6666666666666666)

def test_metrics_f1_score_multiclass(multiclass_data):
    predictions, targets = multiclass_data
    metrics = ClassificationMetrics(predictions, targets, average="macro")
    f1_score = metrics.calculate_f1_score()
    assert f1_score == pytest.approx(0.6666666666666666)