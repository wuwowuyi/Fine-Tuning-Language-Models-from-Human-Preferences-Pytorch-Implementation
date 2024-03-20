import os

import numpy as np
import torch
from torch.utils.data import DataLoader

from lm_human_preferences.language import lm_datasets as datasets
from lm_human_preferences.params import TaskHParams
from lm_human_preferences.utils import core_torch as utils


#returns a postprocessing function
#it is applied to responses before they are scored
#central example: replace all tokens after truncate_token with padding_token
def postprocess_fn_from_hparams(hparams: TaskHParams, padding_token: int):
    def get_mask(responses: torch.Tensor, truncate_token: int, truncate_after: int):
        # We want to truncate at the first occurrence of truncate_token that appears at or after
        # position truncate_after in the responses
        mask = torch.eq(responses, truncate_token).int()
        mask = torch.cat((torch.zeros_like(mask)[:, :truncate_after], mask[:, truncate_after:]), dim=1)
        return torch.cumsum(mask, dim=1) - mask

    if hparams.truncate_token is not None: # truncate tokens are like '.', '\n'
        def truncate(responses):
            # every pos in mask before and at the first truncate_token is zero, and after at least 1.
            mask = get_mask(responses, hparams.truncate_token, hparams.truncate_after)
            return torch.where(mask.bool(), padding_token * torch.ones_like(responses), responses)
        return truncate
    else:
        return lambda responses: responses

#returns a filter function
#responses not passing that function will receive a low (fixed) score
#only query humans on responses that pass that function
#central example: ensure that the sample contains truncate_token
def filter_fn_from_hparams(hparams: TaskHParams):
    def filter(responses):
        if hparams.truncate_token is not None:
            # we prefer a truncate_token after truncate_after in response
            matches_token = torch.eq(responses[:, hparams.truncate_after:], hparams.truncate_token)
            return torch.any(matches_token, dim=-1)
        else:
            return torch.ones(responses.shape[0], dtype=torch.bool)
    return filter


def query_formatter(hparams: TaskHParams, encoder):
    """Turns a query into a context to feed to the language model

    NOTE: Both of these are lists of tokens
    """
    def query_formatter(queries: torch.Tensor):
        batch_size = queries.shape[0]
        device = utils.get_device(queries)
        prefix_tokens = torch.as_tensor(encoder.encode(hparams.query_prefix), dtype=torch.int32, device=device)
        tiled_prefix = torch.tile(prefix_tokens[None], (batch_size, 1))
        suffix_tokens = torch.as_tensor(encoder.encode(hparams.query_suffix), dtype=torch.int32, device=device)
        tiled_suffix = torch.tile(suffix_tokens[None], (batch_size, 1))
        return torch.cat([tiled_prefix, queries, tiled_suffix], 1)
    return query_formatter


def make_query_sampler(*, hparams: TaskHParams, encoder, batch_size: int, mode='train', device: str):
    if hparams.start_text:
        start_token, = encoder.encode(hparams.start_text)
    else:
        start_token = None

    if hparams.end_text:
        end_token, = encoder.encode(hparams.end_text)
    else:
        end_token = None

    data_sampler = datasets.get_dataset(hparams.query_dataset).tf_dataset(
        sequence_length=hparams.query_length, batch_size=batch_size, mode=mode, encoder=encoder,
        start_token=start_token, end_token=end_token,
    )

    def sampler():
        return torch.as_tensor(data_sampler(), dtype=torch.int32, device=device)
    return sampler
