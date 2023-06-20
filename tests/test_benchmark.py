from typing import Callable, List, Tuple, Dict
import pytest
from src.benchmark import Benchmark, CustomBenchmark
from src.dataset import DatasetBase, DatasetInfo, DatasetsList, CustomDataset, BenchmarkData

def callback(ds: DatasetInfo) -> List[float]:
    return [0.55] * len(ds.data['label'])

def test_with_bert_model():
    benchmark = Benchmark(DatasetsList.SST, callback, reload_cache=False)
    metrics_results = benchmark.run(metrics=['mae', 'mse', 'rmse', 'r2_score'])
    assert (metrics_results == [('mae', 0.22279482143072513), ('mse', 0.06464221059971843), ('rmse', 0.2542483246743593), ('r2_score', -0.025113340523399863)])
    assert(benchmark.dataset_info.dataset_link == "https://huggingface.co/datasets/sst")
    assert(benchmark.dataset_info.dataset_info_link == "https://paperswithcode.com/dataset/sst")