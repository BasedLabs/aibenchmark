import logging
import os
from abc import ABC, abstractmethod
from enum import Enum
import gdown

from src.decompress import decompress_7z
from src.exceptions import NotSupportableDecompressionFileFormat

datasets_download_dir = 'datasets_downloads'

if not os.path.isdir(datasets_download_dir):
    os.makedirs(datasets_download_dir)
    logging.info(f'Created download dir {datasets_download_dir}')
else:
    logging.info(f'Download dir {datasets_download_dir} exists, skipping creating')


class DatasetInfo:
    def __init__(self, output_format: str, input_format: str, data: any):
        self.output_format = output_format
        self.input_format = input_format
        self.data = data


class DatasetBase(ABC):
    @abstractmethod
    def load(self, reload_cache: bool = False) -> DatasetInfo:
        pass


class GoogleDriveDataset(DatasetBase, ABC):
    def __init__(self, file_name: str, url: str):
        self.url = url
        if not file_name.endswith('.7z'):
            raise NotSupportableDecompressionFileFormat()
        self.file_name = file_name
        self.download_folder_name = os.path.join(datasets_download_dir, file_name.rstrip('.7z'))
        self.download_file_path = os.path.join(datasets_download_dir, file_name)

    @abstractmethod
    def load(self, reload_cache: bool = False) -> DatasetInfo:
        pass

    def download_from_google_drive(self, download_file_path: str):
        logging.info(f'Loading dataset {self.file_name} from google drive')
        gdown.download(url=self.url, output=download_file_path, quiet=False)
        logging.info(f'Loading dataset {self.file_name} from google drive completed')


class GoogleDriveTestDataset(GoogleDriveDataset):
    def _read_images_from_directory(self, directory):
        files = os.listdir(directory)
        images = [open(os.path.join(directory, file), 'rb').read() for file in files if file.endswith('.jpg')]
        return images

    def load(self, reload_cache=False) -> DatasetInfo:
        if reload_cache:
            self.download_from_google_drive(download_file_path=self.download_file_path)
            logging.info(f'Decomperssing dataset file from {self.download_file_path} to folder {self.download_folder_name}')
            decompress_7z(file_path=self.download_file_path, destination_folder_path=self.download_folder_name)
        images = self._read_images_from_directory(self.download_folder_name)
        return DatasetInfo('list of bytes objects (.jpg images)', 'list of integers in range [0, 1]', images)


class PytorchDataset(DatasetBase):
    def __init__(self, name: str):
        self.name = name

    def load(self, reload_cache: bool = False) -> DatasetInfo:
        pass


class DatasetEnum(Enum):
    GOOGLE_DRIVE_DATASET: DatasetBase = GoogleDriveTestDataset(
        'half-life--wallpaper.7z',
        "https://drive.google.com/uc?id=1D9gbIfv7aAWq1i_7IqqloU9NI6ZYuxlh")
    PYTORCH_DEFAULT_DATASET: DatasetBase = PytorchDataset("test")  # pytorch class
