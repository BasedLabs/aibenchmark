from typing import Callable

from src.datasets import Dataset


class Benchmark:
    def __init__(self, dataset: Dataset, model: any, preprocess_func: Callable, postprocess_func: Callable):
        pass

    @property
    def output_format(self):
        return {}

    @property
    def input_format(self):
        return {}

    def run(self):
        pass