import logging
from benchmark import CustomBenchmark, Benchmark, DatasetBase, DatasetInfo, CustomDataset, BenchmarkData
from metrics import RegressionMetrics, ClassificationMetrics

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
