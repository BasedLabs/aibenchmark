import pandas as pd
import torch
import numpy as np

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, \
    mean_absolute_error, mean_squared_error, r2_score
from sklearn.base import BaseEstimator

from keras import Model as KerasModel

from collections import Counter
from typing import List
import math
import re

class BaseMetricsClass:
    def __init__(self, predictions, targets):
        self.predictions = predictions
        self.targets = targets

    def _convert_to_numpy(self, data):
        #if isinstance(data, (torch.utils.data.Dataset, tf.data.Dataset)):
        #    data = list(data)
        #if isinstance(data, (torch.Tensor, tf.Tensor)):
        #    data = data.numpy()
        if isinstance(data, list):
            data = np.array(data)
        elif isinstance(data, pd.DataFrame):
            data = data.values
        return data

    def count_model_params(self, model):
        if isinstance(model, BaseEstimator):
            return sum([param.size for param in model.get_params().values()])
        elif isinstance(model, KerasModel):
            return model.count_params()
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
        targets = self._convert_to_numpy(self.targets)
        return accuracy_score(targets, predictions)

    def calculate_precision(self):
        predictions = self._convert_to_numpy(self.predictions)
        targets = self._convert_to_numpy(self.targets)
        return precision_score(targets, predictions, average=self.average)

    def calculate_recall(self):
        predictions = self._convert_to_numpy(self.predictions)
        targets = self._convert_to_numpy(self.targets)
        return recall_score(targets, predictions, average=self.average)

    def calculate_f1_score(self):
        predictions = self._convert_to_numpy(self.predictions)
        targets = self._convert_to_numpy(self.targets)
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
    def __init__(self, references: List[str], tokenizer=None):
        self.references = references
        self.tokenizer = tokenizer or self.default_tokenizer

    @staticmethod
    def default_tokenizer(text):
        return re.findall(r'\w+', text.lower())

    def compute_perplexity(self, text: str, tokenizer=None) -> float:
        tokens = tokenizer or self.tokenizer
        tokenized_text = tokens(text)
        token_counts = Counter(tokenized_text)
        total_tokens = sum(token_counts.values())
        perplexity = math.exp(-sum(math.log(token_counts[token] / total_tokens) for token in token_counts) / total_tokens)
        return perplexity

    def compute_bleu(self, text: str, tokenizer=None) -> float:
        tokens = tokenizer or self.tokenizer
        candidate_tokens = tokens(text)
        reference_tokens = [tokens(ref) for ref in self.references]
        reference_token_counts = [Counter(ref) for ref in reference_tokens]
        candidate_token_counts = Counter(candidate_tokens)

        clipped_counts = {token: min(candidate_token_counts[token], max(ref_counts[token] for ref_counts in reference_token_counts)) for token in candidate_token_counts}
        total_candidate_tokens = sum(candidate_token_counts.values())
        total_clipped_counts = sum(clipped_counts.values())

        bleu_score = total_clipped_counts / total_candidate_tokens if total_candidate_tokens != 0 else 0.0
        return bleu_score

    def compute_rouge(self, text: str, tokenizer=None) -> float:
        tokens = tokenizer or self.tokenizer
        candidate_tokens = tokens(text)
        reference_tokens = [tokens(ref) for ref in self.references]
        reference_ngrams = [self.get_ngrams(ref_tokens) for ref_tokens in reference_tokens]
        candidate_ngrams = self.get_ngrams(candidate_tokens)

        intersection_count = sum(len(candidate_ngrams & ref_ngrams) for ref_ngrams in reference_ngrams)
        reference_count = sum(len(ref_ngrams) for ref_ngrams in reference_ngrams)

        rouge_score = intersection_count / reference_count if reference_count != 0 else 0.0
        return rouge_score

    @staticmethod
    def get_ngrams(tokens, n=1):
        return set(tuple(tokens[i:i+n]) for i in range(len(tokens)-n+1))
