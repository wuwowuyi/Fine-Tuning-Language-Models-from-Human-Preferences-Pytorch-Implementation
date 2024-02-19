"""Interface and implementations of label types for a reward model."""

from abc import ABC, abstractmethod
from typing import Optional

import numpy as np
import torch
from torch.nn import functional as F

from lm_human_preferences.utils.core_torch import Schema, pearson_r


class LabelType(ABC):
    @abstractmethod
    def label_schemas(self) -> dict[str, Schema]:
        """Schema for the human annotations."""

    @abstractmethod
    def target_scales(self, labels: dict[str, torch.Tensor]) -> Optional[torch.Tensor]:
        """Extracts scalars out of labels whose scale corresponds to the reward model's output.
           May be none if the labels have no such information."""

    @abstractmethod
    def loss(self, reward_model, labels: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        """
        :param labels: the questions with their labels
        :returns: a dict of stats, including 'loss' for the actual loss
        """

    @abstractmethod
    def question_schemas(self, *, query_length, response_length) -> dict[str, Schema]:
        """Schema for the questions associated with this LabelType."""


class PickBest(LabelType):
    """Pick best response amongst N."""
    def __init__(self, num_responses):
        self.num_responses = num_responses

    def label_schemas(self):
        return dict(best=Schema(torch.int32, ()))

    def target_scales(self, labels):
        return None

    def loss(self, reward_model, labels: torch.Tensor):
        logits = torch.stack([reward_model(labels['query'], labels[f'sample{i}'])
                         for i in range(self.num_responses)], dim=1)  # shape=(b, num_responses)
        error = F.cross_entropy(logits, target=labels['best'].long())
        return dict(loss=error, error=error)

    def question_schemas(self, *, query_length, response_length) -> dict[str, Schema]:
        return dict(
            query=Schema(torch.int32, (query_length,)),
            **{f"sample{i}": Schema(torch.int32, (response_length,)) for i in range(self.num_responses)}
        )


class ScalarRating(LabelType):
    """Rate a single number with a scalar score."""
    def __init__(self):
        pass

    def label_schemas(self):
        return dict(
            score=Schema(torch.float, ()))

    def target_scales(self, labels):
        return labels['score']

    def loss(self, reward_model, labels):
        predicted = reward_model(labels['query'], labels['sample'])
        labels = labels['score']
        error = torch.mean((labels - predicted) ** 2, dim=0)
        label_var, label_mean = torch.var_mean(labels, dim=0, correction=0)  # tensorflow has no Bessel's correction
        corr = pearson_r(labels, predicted)
        return dict(loss=error, error=error,
                    label_mean=label_mean, label_var=label_var, corr=corr)

    def question_schemas(self, *, query_length, response_length) -> dict[str, Schema]:
        return dict(
            query=Schema(torch.int32, (query_length,)),
            sample=Schema(torch.int32, (response_length,)),
        )


class ScalarComparison(LabelType):
    """Give a scalar indicating difference between two responses."""
    def label_schemas(self):
        return dict(difference=Schema(torch.float, ()))

    def target_scales(self, labels):
        # Divide by two to get something with the same variance as the trained reward model output
        return labels['difference']/2

    def loss(self, reward_model, labels):
        outputs0 = reward_model(labels['query'], labels['sample0'])
        outputs1 = reward_model(labels['query'], labels['sample1'])

        differences = labels['difference']
        predicted_differences = outputs1 - outputs0
        error = torch.mean((differences - predicted_differences)**2, dim=0)
        return dict(loss=error, error=error)

    def question_schemas(self, *, query_length, response_length) -> dict[str, Schema]:
        return dict(
            query=Schema(torch.int32, (query_length,)),
            sample0=Schema(torch.int32, (response_length,)),
            sample1=Schema(torch.int32, (response_length,)),
        )


def get(label_type: str) -> LabelType:
    if label_type == 'scalar_rating':
        return ScalarRating()
    if label_type == 'scalar_compare':
        return ScalarComparison()
    if label_type.startswith('best_of_'):
        n = int(label_type[len('best_of_'):])
        return PickBest(n)
    raise ValueError(f"Unexpected label type {label_type}")
