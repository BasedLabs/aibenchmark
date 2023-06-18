import logging
from typing import Callable, Union, List, Tuple

from src.datasets import DatasetBase, DatasetInfo, DatasetEnum
from src.metrics import RegressionMetrics, ClassificationMetrics
from src.exceptions import NotSupportedTask


class Benchmark:
    def __init__(self, dataset: DatasetBase, callback: Callable[[DatasetInfo], any]):
        self._dataset: DatasetBase = dataset
        self._callback = callback
        self._dataset_info: Union[DatasetInfo, None] = None

    @staticmethod
    def load(dataset: DatasetEnum, callback: Callable[[DatasetInfo], any], reload_cache: bool=False) -> 'Benchmark':
        '''
        Load a specified dataset
        :param dataset: instance of DatasetBase. Use DatasetEnum
        :param callback:
        :return:
        '''
        benchmark = Benchmark(dataset=dataset.value,
                              callback=callback)
        logging.info(f'Initializing dataset {dataset}')
        if reload_cache:
            logging.info('reload_cache parameter is enabled. This will reload dataset from server')
        benchmark._dataset_info = benchmark._dataset.load(reload_cache=reload_cache)
        return benchmark

    @property
    def output_format(self):
        return self._dataset_info.output_format

    @property
    def input_format(self):
        return self._dataset_info.input_format

    def get_existing_benchmarks(self) -> Dict[Str, List[Tuple[str, int]]]:
        pass

    def run(self, task: str, metrics: List[str], custom_metric = None):
        """
        :task: ['regression', 'classification']
        :metrics: Available classification metrics: ['accuracy', 'precision', 'recall', 'f1_score']
                  Available regression metrics: ['mae', 'mse', 'rmse', 'r2']
        :custom_metric: your python function to calculate a metric of your preference
        """
        predictions = self._callback(self._dataset_info)
        targets = self._dataset.targets
        metric_calculator = None

        if task == 'regression':
            metric_calculator = RegressionMetrics(predictions, targets)
        elif task  == 'classification':
            metric_calculator = ClassificationMetrics(predictions, targets)
        else:
            raise NotSupportedTask('This kind of task is not currently supported, use Benchmark.run_custom(data, metric_function)')


        results = []
        for metric in metrics:
            results.append((metric, metric_calculator.calculate_metric(metric))

        if custom_metric:
            results.append((custom_metric.__name__, custom_metric(predictions, targets)))

        return results

    
class CustomBenchmark:
    """
    Benchmark on a custom dataset on a single or multiple models
    """

    def __init__(self, features: List[any], targets: List[any]):
        self._features = features
        self.targets = targets

    def run(model: any, metrics: List[str], custom_metric = None):
        """
        :task: ['regression', 'classification']
        :metrics: Available classification metrics: ['accuracy', 'precision', 'recall', 'f1_score']
                  Available regression metrics: ['mae', 'mse', 'rmse', 'r2']
        :custom_metric: your python function to calculate a metric of your preference
        """

        if task == 'regression':
            metric_calculator = RegressionMetrics(predictions, targets)
        elif task  == 'classification':
            metric_calculator = ClassificationMetrics(predictions, targets)
        else:
            raise NotSupportedTask('This kind of task is not currently supported, use Benchmark.run_custom(data, metric_function)')

        results = []
        for metric in metrics:
            results.append((metric, metric_calculator.calculate_metric(metric))

        if custom_metric:
            results.append((custom_metric.__name__, custom_metric(predictions, targets)))

        return results


if __name__ == '__main__':
    benchmark = Benchmark.load(DatasetEnum.CIFAR10, lambda x: print(x), reload_cache=True)
    print(benchmark.output_format)
