from typing import Callable, List, Tuple, Dict
import pytest
from src.benchmark import Benchmark, CustomBenchmark
from src.dataset import DatasetBase, DatasetInfo, DatasetsList, CustomDataset, BenchmarkData

def callback(ds: DatasetInfo) -> List[float]:
    return [0.55] * len(ds.data['label'])

def test_benchmark():
    benchmark = Benchmark(DatasetsList.Texts.SST, callback, reload_cache=True)
    metrics_results = benchmark.run(metrics=['mae', 'mse', 'rmse', 'r2_score'])
    assert (metrics_results == [('mae', 0.23090618396776294), ('mse', 0.06723232037407614), ('rmse', 0.2592919597173737), ('r2_score', -0.022610548165998834)])
    assert(benchmark.dataset_info.dataset_link == "https://huggingface.co/datasets/sst")
    assert(benchmark.dataset_info.dataset_info_link == "https://paperswithcode.com/dataset/sst")