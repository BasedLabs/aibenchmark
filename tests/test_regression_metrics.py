import pytest
import torch
from aibenchmark.metrics import RegressionMetrics

@pytest.fixture
def regression_data():
    predictions = torch.tensor([1.2, 2.4, 3.1, 4.5, 5.0])
    targets = torch.tensor([1.0, 2.5, 2.9, 4.8, 4.7])
    yield predictions, targets

def test_regression_metrics_mae(regression_data):
    predictions, targets = regression_data
    metrics = RegressionMetrics(predictions, targets)
    mae = metrics.calculate_mae()
    assert mae == pytest.approx(0.22000003)

def test_regression_metrics_mse(regression_data):
    predictions, targets = regression_data
    metrics = RegressionMetrics(predictions, targets)
    mse = metrics.calculate_mse()
    assert mse == pytest.approx(0.05400003)

def test_regression_metrics_rmse(regression_data):
    predictions, targets = regression_data
    metrics = RegressionMetrics(predictions, targets)
    rmse = metrics.calculate_rmse()
    assert rmse == pytest.approx(0.23237906)

def test_regression_metrics_r2_score(regression_data):
    predictions, targets = regression_data
    metrics = RegressionMetrics(predictions, targets)
    r2_score = metrics.calculate_r2_score()
    assert r2_score == pytest.approx(0.9736018624746676)