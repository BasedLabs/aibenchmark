import logging
from aibenchmark.benchmark import CustomBenchmark, Benchmark, DatasetBase, DatasetInfo, CustomDataset, BenchmarkData
from aibenchmark.metrics import RegressionMetrics, ClassificationMetrics

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
