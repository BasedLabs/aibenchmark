from abc import ABC, abstractmethod
from enum import Enum


class Descriptor:
    def __init__(self, output_format, input_format, data):
        self.output_format = output_format
        self.input_format = input_format
        self.data = data


class DatasetBase(ABC, Enum):
    GOOGLE_DRIVE_DATASET = Dataset("path to google drive here")
    PYTORCH_DEFAULT_DATASET = Dataset()  # pytorch class

    @abstractmethod
    def load(self) -> Descriptor:
        pass


class GoogleDriveDataset(DatasetBase):
    def __init__(self, url: str):
        self.url = url

    def load(self) -> Descriptor:
        pass


class PytorchDataset(DatasetBase):
    def __init__(self, name: str):
        self.name = name

    def load(self) -> Descriptor:
        pass


