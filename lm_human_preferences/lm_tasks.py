from dataclasses import dataclass, field
from typing import Optional

import torch
from torch.utils.data import DataLoader

from lm_human_preferences.language import lm_datasets as datasets
from lm_human_preferences.utils import hyperparams


@dataclass
class PolicyHParams(hyperparams.HParams):
    temperature: float = 1.0
    initial_model: str = None

@dataclass
class TaskHParams(hyperparams.HParams):
    # Query params
    query_length: int = None
    query_dataset: str = None
    query_prefix: str = ''
    query_suffix: str = ''
    start_text: Optional[str] = '.'
    end_text: Optional[str] = None

    # Response params
    response_length: int = None

    # Truncate response after the first occurrence of this token at or after index after when sampling.
    truncate_token: Optional[int] = None
    truncate_after: int = 0
    penalty_reward_value: int = -1

    policy: PolicyHParams = field(default_factory=PolicyHParams)

#returns a postprocessing function
#it is applied to responses before they are scored
#central example: replace all tokens after truncate_token with padding_token
def postprocess_fn_from_hparams(hparams: TaskHParams, padding_token: int):
    def get_mask(responses, truncate_token, truncate_after):
        # We want to truncate at the first occurrence of truncate_token that appears at or after
        # position truncate_after in the responses
        mask = tf.cast(tf.equal(responses, truncate_token), tf.int32)
        mask = tf.concat([tf.zeros_like(mask)[:,:truncate_after], mask[:,truncate_after:]], axis=1)
        return tf.cast(tf.cumsum(mask, axis=1) - mask, tf.bool)
    if hparams.truncate_token is not None:
        def truncate(responses):
            mask = get_mask(responses, hparams.truncate_token, hparams.truncate_after)
            return tf.compat.v1.where(mask, padding_token * tf.ones_like(responses), responses)
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
            matches_token = tf.equal(responses[:, hparams.truncate_after:], hparams.truncate_token)
            return tf.reduce_any(matches_token, axis=-1)
        else:
            return tf.ones(tf.shape(responses)[0], dtype=tf.bool)
    return filter


def query_formatter(hparams: TaskHParams, encoder):
    """Turns a query into a context to feed to the language model

    NOTE: Both of these are lists of tokens
    """
    def query_formatter(queries):
        batch_size = queries.shape[0]
        prefix_tokens = torch.as_tensor(encoder.encode(hparams.query_prefix), dtype=torch.int32)
        tiled_prefix = torch.tile(prefix_tokens[None], (batch_size, 1))
        suffix_tokens = torch.as_tensor(encoder.encode(hparams.query_suffix), dtype=torch.int32)
        tiled_suffix = torch.tile(suffix_tokens[None], (batch_size, 1))
        return torch.cat([tiled_prefix, queries, tiled_suffix], 1)
    return query_formatter


def make_query_sampler(*, hparams: TaskHParams, encoder, batch_size: int, mode='train'):
    if hparams.start_text:
        start_token, = encoder.encode(hparams.start_text)
    else:
        start_token = None

    if hparams.end_text:
        end_token, = encoder.encode(hparams.end_text)
    else:
        end_token = None

    # NOTE: MPI not supported here. can add support of DDP later.
    data = datasets.get_dataset(hparams.query_dataset).tf_dataset(
        sequence_length=hparams.query_length, mode=mode, encoder=encoder,
        start_token=start_token, end_token=end_token,
    )
    loader = DataLoader(data.with_format("torch"), batch_size, drop_last=True,
                        pin_memory=True, num_workers=1)
    data_iter = iter(loader)

    def sampler(scope=None):
        context_tokens = torch.as_tensor(next(data_iter), dtype=torch.int32)
        return dict(tokens=context_tokens)
    return sampler
