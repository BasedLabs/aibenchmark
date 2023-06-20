import json
import logging
import os
from abc import ABC, abstractmethod
from typing import List, Iterable

import gdown
from datasets import load_dataset

from src.exceptions import TestDataNotFoundInHuggingFaceDataset

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
    def __init__(self,
                 dataset_format: any,
                 data: any,
                 ground_truth: any,
                 benchmarks_data: List[BenchmarkData]):
        self.dataset_format = dataset_format
        self.data = data
        self.ground_truth = ground_truth
        self.benchmarks_data = benchmarks_data


class DatasetBase(ABC):
    @abstractmethod
    def load(self, reload_cache: bool = False) -> DatasetInfo:
        pass


class CustomDataset(DatasetBase):
    def __init__(self, ground_truth: Iterable):
        self._ground_truth = ground_truth

    def load(self, reload_cache: bool = False) -> DatasetInfo:
        dataset_info = DatasetInfo(
            dataset_format='custom dataset',
            data=[],
            ground_truth=self._ground_truth,
            benchmarks_data=[]
        )

        return dataset_info


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
        file_id = self.url.replace('https://drive.google.com/file/d/','')
        file_id = file_id.replace('/view?usp=drive_link', '')
        file_id = file_id.replace('/view?usp=sharing', '')
        download_url = f'https://drive.google.com/uc?id={file_id}'

        gdown.download(url=download_url, output=download_file_path, quiet=False)
        logging.info(f'Loading dataset {self.file_name} from google drive completed')


class PapersWithCodeHuggingFaceDataset(GoogleDriveDataset):
    def __init__(self, url, file_name: str):
        super().__init__(file_name, url)

    def _get_test_data(self, huggingface_dataset):
        keys = ['test', 'validation']
        for key in keys:
            if key in huggingface_dataset:
                return huggingface_dataset[key]

        raise TestDataNotFoundInHuggingFaceDataset()

    def _get_ground_truth_data(self, huggingface_test_data):
        keys = ['label', 'labels', 'feature']
        for key in keys:
            if key in huggingface_test_data.data.column_names:
                return huggingface_test_data[key]

        raise TestDataNotFoundInHuggingFaceDataset()

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
        dataset = load_dataset(huggingface_dataset_name,)
        test_data = self._get_test_data(dataset)
        ground_truth = self._get_ground_truth_data(test_data)
        dataset_info = DatasetInfo(dataset_format=test_data.features,
                                   data=test_data,
                                   ground_truth=ground_truth,
                                   benchmarks_data=benchmarks)
        return dataset_info


class TextsDatasetsList:
    SST = PapersWithCodeHuggingFaceDataset(
        url='https://drive.google.com/file/d/1jvlSyfom_0oBd1D2PtZDjab7waX_WwxE/view?usp=drive_link',
        file_name="1-2-dataset-sst.json")
    MultiNli = PapersWithCodeHuggingFaceDataset(
        url='https://drive.google.com/file/d/1n-Oqd5tCm-X12A1aB5u2nXMrifb5NicP/view?usp=sharing',
        file_name='1-3-dataset-multinli'
    )
    ImdbMovieReviews = PapersWithCodeHuggingFaceDataset(
        url='https://drive.google.com/file/d/1Yn1h9ECVbyM15qMfpU6mpJqFqIQWRjyQ/view?usp=drive_link',
        file_name='1-4-dataset-imdb-movie-reviews.json'
    )
    SNLI = PapersWithCodeHuggingFaceDataset(
        url='https://drive.google.com/file/d/1_1EWLr-snR9bi8KEo6-2HEnkPohdcDHD/view?usp=drive_link',
        file_name='1-7-dataset-snli.json'
    )
    NaturalQuestions = PapersWithCodeHuggingFaceDataset(
        url='https://drive.google.com/file/d/1V8-bqnbhdGl9uvNUdzG7PvRojZCTAsnV/view?usp=drive_link',
        file_name='1-10-dataset-natural-questions.json'
    )
    ConceptNet = PapersWithCodeHuggingFaceDataset(
        url='https://drive.google.com/file/d/1FkzOvqjJsdDGVwfgatoNZbcgvwGAtfnr/view?usp=drive_link',
        file_name='1-11-dataset-conceptnet.json'
    )
    AgNews = PapersWithCodeHuggingFaceDataset(
        url='https://drive.google.com/file/d/11K79pV4VPFb5FtTWYoPmCtDM8AiI2iLF/view?usp=drive_link',
        file_name='1-15-dataset-ag-news.json'
    )
    CONLL2003 = PapersWithCodeHuggingFaceDataset(
        url='https://drive.google.com/file/d/1bo73BCo-jZ3hLjHhJNVw1tGOQLNGabko/view?usp=drive_link',
        file_name='1-17-dataset-conll-2003.json',
    )
    DbPedia = PapersWithCodeHuggingFaceDataset(
        url='https://drive.google.com/file/d/1g2LDksBtGS9WtLpt7_6UKHGwbVcrU4bX/view?usp=drive_link',
        file_name='1-19-dataset-dbpedia.json'
    )
    CNNDailyMail1 = PapersWithCodeHuggingFaceDataset(
        url='https://drive.google.com/file/d/1vLjOa9W11u4FZyVh01d6V5QmWChpjqT-/view?usp=drive_link',
        file_name='1-24-dataset-cnn-daily-mail-1.json'
    )
    DailyDialog = PapersWithCodeHuggingFaceDataset(
        url='https://drive.google.com/file/d/1NfwcijXzsU0tsviiPefrXYiiomOPLm-Q/view?usp=drive_link',
        file_name='1-31-dataset-dailydialog.json'
    )
    VCTK = PapersWithCodeHuggingFaceDataset(
        url='https://drive.google.com/file/d/1pM010S3MGFuUzCFVO4zFliUStlwAncux/view?usp=drive_link',
        file_name='1-34-dataset-vctk.json'
    )
    MultiWOZ = PapersWithCodeHuggingFaceDataset(
        url='https://drive.google.com/file/d/1ai0Qa5UASc1wsG-kuvmKSwoS91sUUpl2/view?usp=drive_link',
        file_name='1-35-dataset-multiwoz.json'
    )
    ConceptualCaptions = PapersWithCodeHuggingFaceDataset(
        url='https://drive.google.com/file/d/1MFHP9dSZPPNIRFZVyPRGsuIGgJyQNO5Q/view?usp=drive_link',
        file_name='1-39-dataset-conceptual-captions.json'
    )


class ImagesDatasetsList:
    CIFAR10 = PapersWithCodeHuggingFaceDataset(
        url='https://drive.google.com/file/d/1tj42UBqxQkc9SvN39ryGDsgQxlvasXqe/view?usp=drive_link',
        file_name='1-0-dataset-cifar-10.json'
    )
    COCO = PapersWithCodeHuggingFaceDataset(
        url='https://drive.google.com/file/d/1MydcQlrXjfxjhMwFOEiA_rAB5Xg4W1Vz/view?usp=drive_link',
        file_name='1-2-dataset-coco.json'
    )


class DatasetsList:
    Texts: TextsDatasetsList = TextsDatasetsList
    Images: ImagesDatasetsList = ImagesDatasetsList

    @staticmethod
    def get_available_datasets():
        for key in DatasetsList.Texts.__dict__:
            if '__' not in key: # builtin method
                yield 'DatasetsList.Texts.' + key
        for key in DatasetsList.Images.__dict__:
            if '__' not in key: # builtin method
                yield 'DatasetsList.Images.' + key
