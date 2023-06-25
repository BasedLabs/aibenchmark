import pandas as pd
import torch
import torch.nn.functional as F
import numpy as np
import nltk
from nltk.translate.bleu_score import corpus_bleu, sentence_bleu, SmoothingFunction

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, \
    mean_absolute_error, mean_squared_error, r2_score, log_loss
from sklearn.base import BaseEstimator
from sklearn.utils.extmath import softmax

from collections import Counter
from typing import List
from itertools import chain
import math
import re

class BaseMetricsClass:
    def __init__(self, predictions, targets):
        self.predictions = predictions
        self.targets = targets

    def _convert_to_numpy(self, data):
        if isinstance(data, list):
            data = np.array(data)
        elif isinstance(data, pd.DataFrame):
            data = data.values
        return data

    def count_model_params(self, model):
        if isinstance(model, BaseEstimator):
            return sum([param.size for param in model.get_params().values()])
        elif isinstance(model, torch.nn.Module):
            return sum(p.numel() for p in model.parameters())
        else:
            raise ValueError("Unsupported model type.")

    def calculate_metric(self):
        return NotImplementedError


class ClassificationMetrics(BaseMetricsClass):
    def __init__(self, predictions, targets, average='binary'):
        super().__init__(predictions, targets)
        self.average = average

    def calculate_accuracy(self):
        predictions = self._convert_to_numpy(self.predictions)
        targets = self._convert_to_numpy(self.targets).round()
        return accuracy_score(targets, predictions)

    def calculate_precision(self):
        predictions = self._convert_to_numpy(self.predictions)
        targets = self._convert_to_numpy(self.targets).round()
        return precision_score(targets, predictions, average=self.average)

    def calculate_recall(self):
        predictions = self._convert_to_numpy(self.predictions)
        targets = self._convert_to_numpy(self.targets).round()
        return recall_score(targets, predictions, average=self.average)

    def calculate_f1_score(self):
        predictions = self._convert_to_numpy(self.predictions)
        targets = self._convert_to_numpy(self.targets).round()
        return f1_score(targets, predictions, average=self.average)

    def calculate_metric(self, metric_name):
        metric_name = metric_name.lower()
        if metric_name == 'accuracy':
            return self.calculate_accuracy()
        elif metric_name == 'precision':
            return self.calculate_precision()
        elif metric_name == 'recall':
            return self.calculate_recall()
        elif metric_name == 'f1_score':
            return self.calculate_f1_score()
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


class NLPMetrics:
    def __init__(self):
        pass

    def compute_perplexity(self, predicted_logits: List[float], target_ids: List[int]) -> float:
        """
        :predicted_logits: - list of model predictions for each text (probability distribution over vocabulary)
        "target_ids" - ground truth of text ids for each text

        i.e.

        if you have two texts [["I", "love", "dogs"], ["I", "like", "Cats"]]
        with a vocabulary {0: "I", 1: "like", 2: "love", 3: "Cats", 4: "Dogs"}

        then target_ids are:
        target_ids = [[1, 0, 1, 0, 1], [1, 1, 0, 1, 0]]

        and the predicted_logits should be something like
        predicted_logits = [[0.87, 0.52, 0.7, 0.12, 0.89], [0.88, 0.74, 0.51, 0.9, 0.23]]
        """
        # Convert predicted logits to probabilities
        predicted_probs = F.softmax(torch.from_numpy(np.asarray(predicted_logits)), dim=-1)

        # Flatten the predicted probabilities and target IDs
        predicted_probs = predicted_probs.view(-1, predicted_probs.size(-1))
        target_ids = torch.from_numpy(np.asarray(target_ids)).float()

        # Calculate cross-entropy loss
        cross_entropy_loss = F.cross_entropy(predicted_probs, target_ids, reduction='none')

        # Calculate perplexity
        perplexity = torch.exp(torch.mean(cross_entropy_loss))

        return perplexity.item()

    @staticmethod
    def get_ngrams(tokens, n=1):
        return set(tuple(tokens[i:i+n]) for i in range(len(tokens)-n+1))
