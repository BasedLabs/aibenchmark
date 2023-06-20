from typing import Callable, List, Tuple, Dict
import pytest
from src.benchmark import Benchmark, CustomBenchmark
from src.dataset import DatasetBase, DatasetInfo, DatasetsList, CustomDataset, BenchmarkData

def sst_callback(ds: DatasetInfo) -> List[float]:
    return [0.55] * len(ds.data['label'])

def test_sst_benchmark():
    benchmark = Benchmark(DatasetsList.Texts.SST, sst_callback, reload_cache=True)
    metrics_results = benchmark.run(metrics=['mae', 'mse', 'rmse', 'r2_score'])
    assert (metrics_results == [('mae', 0.23090618396776294), ('mse', 0.06723232037407614), ('rmse', 0.2592919597173737), ('r2_score', -0.022610548165998834)])
    assert(benchmark.dataset_info.dataset_link == "https://huggingface.co/datasets/sst")
    assert(benchmark.dataset_info.dataset_info_link == "https://paperswithcode.com/dataset/sst")



def cidar10_callback(ds: DatasetInfo):
    return [1 for _ in range(len(ds.data['label']))]

def test_cifar10_benchmark():
    benchmark = Benchmark(DatasetsList.Images.CIFAR10, cidar10_callback, reload_cache=True)
    metrics_results = benchmark.run(metrics=['accuracy', 'precision', 'recall', 'f1_score'], average='macro')
    assert(metrics_results == [('accuracy', 0.1), ('precision', 0.01), ('recall', 0.1), ('f1_score', 0.01818181818181818)])
