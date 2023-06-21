import pytest

from typing import List
import random

from aibenchmark.benchmark import Benchmark
from aibenchmark.dataset import DatasetInfo, DatasetsList


def sst_regression_callback(dataset: DatasetInfo) -> List[float]:
    return [0.55] * len(dataset.data['label'])

def sst_classification_callback(dataset: DatasetInfo) -> List[float]:
    return [0 for _ in range(len(dataset.data['label']))]

def test_sst_benchmark():
    regression_benchmark = Benchmark(DatasetsList.Texts.SST, sst_regression_callback)
    regression_metrics_results = regression_benchmark.run(metrics=['mae', 'mse', 'rmse', 'r2_score'])
    assert (regression_metrics_results == [('mae', 0.23090618396776294), ('mse', 0.06723232037407614),
                                ('rmse', 0.2592919597173737), ('r2_score', -0.022610548165998834)])
    
    classification_benchmark = Benchmark(DatasetsList.Texts.SST, sst_classification_callback)
    classification_metrics_results = classification_benchmark.run(metrics=['accuracy', 'precision', 'recall', 'f1_score'])
    assert (classification_metrics_results == [('accuracy', 0.5171945701357467), ('precision', 0),
                                ('recall', 0), ('f1_score', 0)])
    assert (classification_benchmark.dataset_info.dataset_link == "https://huggingface.co/datasets/sst")
    assert (classification_benchmark.dataset_info.dataset_info_link == "https://paperswithcode.com/dataset/sst")

def cifar10_callback(dataset: DatasetInfo):
    return [1 for _ in range(len(dataset.data['label']))]

def test_cifar10_benchmark():
    benchmark = Benchmark(DatasetsList.Images.CIFAR10, cifar10_callback)
    metrics_results = benchmark.run(metrics=['accuracy', 'precision', 'recall', 'f1_score'], average='macro')
    assert (metrics_results == [('accuracy', 0.1), ('precision', 0.01), ('recall', 0.1),
                                ('f1_score', 0.01818181818181818)])
