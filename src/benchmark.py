import logging
from typing import Callable, Union

from src.datasets import DatasetBase, DatasetInfo, DatasetEnum


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

    def run(self):
        user_model_result = self._callback(self._dataset_info)
        print(user_model_result)
        # actual benchmark here


if __name__ == '__main__':
    benchmark = Benchmark.load(DatasetEnum.CIFAR10, lambda x: print(x), reload_cache=True)
    print(benchmark.output_format)
