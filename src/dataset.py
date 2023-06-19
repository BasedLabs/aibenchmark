import json
import logging
import os
from abc import ABC, abstractmethod
from enum import Enum
from typing import List

import gdown
from datasets import load_dataset

datasets_download_dir = 'downloads'

if not os.path.isdir(datasets_download_dir):
    os.makedirs(datasets_download_dir)
    logging.info(f'Created download dir {datasets_download_dir}')
else:
    logging.info(f'Download dir {datasets_download_dir} exists, skipping creating')


class BenchmarkData:
    def __init__(self, model_name, benchmark_result, task_name):
        self.model_name = model_name
        self.benchmark_result = benchmark_result
        self.task_name = task_name


class DatasetInfo:
    def __init__(self, dataset_format: any,
                 data: any, targets: any,
                 benchmarks_data: List[BenchmarkData]):
        self.dataset_format = dataset_format
        self.data = data
        self.targets = targets
        self.benchmarks_data = benchmarks_data


class DatasetBase(ABC):
    @abstractmethod
    def load(self, reload_cache: bool = False) -> DatasetInfo:
        pass


class GoogleDriveDataset(DatasetBase, ABC):
    def __init__(self, file_name: str, url: str):
        self.url = url
        self.file_name = file_name
        self.download_folder_name = os.path.join(datasets_download_dir, file_name.split('.')[0])
        self.download_file_path = os.path.join(datasets_download_dir, file_name)

    @abstractmethod
    def load(self, reload_cache: bool = False) -> DatasetInfo:
        pass

    def download_from_google_drive(self, download_file_path: str):
        logging.info(f'Loading dataset {self.file_name} from google drive')
        gdown.download(url=self.url, output=download_file_path, quiet=False)
        logging.info(f'Loading dataset {self.file_name} from google drive completed')


class PapersWithCodeHuggingFaceDataset(GoogleDriveDataset):
    def __init__(self, url, file_name: str):
        super().__init__(file_name, url)

    def load(self, reload_cache: bool = True) -> DatasetInfo:
        logging.info('Loading benchmarks data')
        if reload_cache:
            self.download_from_google_drive(self.download_file_path)
        json_content = json.load(open(self.download_file_path, 'r'))
        benchmarks = []
        huggingface_dataset_name = json_content[0]['hugging_face_dataset_name']
        for i, benchmark in enumerate(json_content):
            for benchmark_data in benchmark['table']:
                benchmarks.append(BenchmarkData(model_name=benchmark_data['method'],
                                                benchmark_result=benchmark_data['raw_metrics'],
                                                task_name=benchmark['sota']['task_name']))
        logging.info('Loading dataset from hugging face')
        dataset = load_dataset(huggingface_dataset_name)
        test_data = dataset['test'] if 'test' in dataset['test'] else dataset['validation']
        targets = test_data.data['label'] if 'label' in test_data.data.column_names else test_data.data['feature']
        dataset_info = DatasetInfo(dataset_format=test_data.features,
                                   data=test_data,
                                   targets=targets,
                                   benchmarks_data=benchmarks)
        return dataset_info


class DatasetEnum(Enum):
    SST = PapersWithCodeHuggingFaceDataset(
        url="https://drive.google.com/uc?id=1jvlSyfom_0oBd1D2PtZDjab7waX_WwxE",
        file_name="dataset-sst.json")
