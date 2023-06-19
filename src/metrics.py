import pandas as pd
import torch
import tensorflow as tf
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, \
    mean_absolute_error, mean_squared_error, r2_score


class BaseMetricsClass:
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

    def calculate_metric(self):
        return NotImplementedError


class ClassificationMetrics(BaseMetricsClass):
    def __init__(self, predictions, targets):
        super().__init__(predictions, targets)

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

    def calculate_metric(self, metric_name, average='binary'):
        metric_name = metric_name.lower()
        if metric_name == 'accuracy':
            return self.calculate_accuracy()
        elif metric_name == 'precision':
            return self.calculate_precision(average=average)
        elif metric_name == 'recall':
            return self.calculate_recall(average=average)
        elif metric_name == 'f1_score':
            return self.calculate_f1_score(average=average)
        else:
            raise ValueError(f"Unsupported metric: {metric_name}")


class RegressionMetrics(BaseMetricsClass):
    def __init__(self, predictions, targets):
        super().__init__(predictions, targets)

    def calculate_mae(self):
        predictions = self._convert_to_numpy(self.predictions)
        targets = self._convert_to_numpy(self.targets)
        return mean_absolute_error(targets, predictions)

    def calculate_mse(self):
        predictions = self._convert_to_numpy(self.predictions)
        targets = self._convert_to_numpy(self.targets)
        return mean_squared_error(targets, predictions)

    def calculate_rmse(self):
        mse = self.calculate_mse()
        return np.sqrt(mse)

    def calculate_r2_score(self):
        predictions = self._convert_to_numpy(self.predictions)
        targets = self._convert_to_numpy(self.targets)
        return r2_score(targets, predictions)

    def calculate_metric(self, metric_name):
        metric_name = metric_name.lower()
        if metric_name == 'mae':
            return self.calculate_mae()
        elif metric_name == 'mse':
            return self.calculate_mse()
        elif metric_name == 'rmse':
            return self.calculate_rmse()
        elif metric_name == 'r2_score':
            return self.calculate_r2_score()
        else:
            raise ValueError(f"Unsupported metric: {metric_name}")
