import torch
import tensorflow as tf
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


class ClassificationMetrics:
    def __init__(self, predictions, targets):
        self.predictions = predictions
        self.targets = targets

    def _convert_to_numpy(self, data):
        if isinstance(data, (torch.utils.data.Dataset, tf.data.Dataset)):
            data = list(data)
        if isinstance(data, (torch.Tensor, tf.Tensor)):
            data = data.numpy()
        elif isinstance(data, list):
            data = np.array(data)
        elif isinstance(data, pd.DataFrame):
            data = data.values
        return data


    def calculate_accuracy(self):
        predictions = self._convert_to_numpy(self.predictions)
        targets = self._convert_to_numpy(self.targets)
        return accuracy_score(targets, predictions)

    def calculate_precision(self, average='binary'):
        predictions = self._convert_to_numpy(self.predictions)
        targets = self._convert_to_numpy(self.targets)
        return precision_score(targets, predictions, average=average)

    def calculate_recall(self, average='binary'):
        predictions = self._convert_to_numpy(self.predictions)
        targets = self._convert_to_numpy(self.targets)
        return recall_score(targets, predictions, average=average)

    def calculate_f1_score(self, average='binary'):
        predictions = self._convert_to_numpy(self.predictions)
        targets = self._convert_to_numpy(self.targets)
        return f1_score(targets, predictions, average=average)