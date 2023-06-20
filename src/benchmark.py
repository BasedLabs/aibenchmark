import logging
from typing import Callable, Union, List, Iterable

from src.dataset import DatasetBase, DatasetInfo, CustomDataset, BenchmarkData
from src.exceptions import NotSupportedMetric
from src.metrics import RegressionMetrics, ClassificationMetrics


class Benchmark:
    def __init__(self, dataset: DatasetBase, callback: Callable[[DatasetInfo], any]):
        self.dataset: DatasetBase = dataset
        self.callback = callback
        logging.info(f'Initializing dataset {dataset}')
        self.dataset_info: Union[DatasetInfo, None] = dataset.load()

    @property
    def dataset_format(self):
        return self.dataset_info.dataset_format

    def get_existing_benchmarks(self) -> List[BenchmarkData]:
        return self.dataset_info.benchmarks_data

    def run(self, metrics: List[str],
            custom_metric_calculator: Callable[[Iterable, Iterable], any] = None,
            average="binary"):
        """
        :task: ['regression', 'classification']
        :metrics: Available classification metrics: ['accuracy', 'precision', 'recall', 'f1_score']
                  Available regression metrics: ['mae', 'mse', 'rmse', 'r2_score']
        :custom_metric: Your python function to calculate a metric of your preference
        :average: For classification, choose one of the available methods {'micro', 'macro', 'samples', 'weighted', 'binary'} or None
        """
        predictions = self.callback(self.dataset_info)
        targets = self.dataset_info.ground_truth

        metrics_with_calculators = {
            'accuracy': ClassificationMetrics(predictions, targets, average=average),
            'precision': ClassificationMetrics(predictions, targets, average=average),
            'recall': ClassificationMetrics(predictions, targets, average=average),
            'f1_score': ClassificationMetrics(predictions, targets, average=average),
            'mae': RegressionMetrics(predictions, targets),
            'mse': RegressionMetrics(predictions, targets),
            'rmse': RegressionMetrics(predictions, targets),
            'r2_score': RegressionMetrics(predictions, targets)
        }

        for metric in metrics:
            if metric not in metrics_with_calculators:
                raise NotSupportedMetric(
                    f'{metric}: This kind of metric is not currently supported, use Benchmark.run_custom(data, metric_function) or run(data, custom_metric))')

        results = []
        for metric in metrics:
            metric_calculator = metrics_with_calculators[metric]
            results.append((metric, metric_calculator.calculate_metric(metric)))

        if custom_metric_calculator:
            results.append((custom_metric_calculator.__name__, custom_metric_calculator(predictions, targets)))

        return results


class CustomBenchmark(Benchmark):
    """
    Benchmark on a custom dataset on a single or multiple models
    """

    def __init__(self, ground_truth: Iterable, predicted: Iterable):
        self.predicted = predicted
        self.ground_truth = ground_truth

        super().__init__(CustomDataset(ground_truth), lambda dataset_info: predicted)
