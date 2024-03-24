from pathlib import Path

import torch

from launch import get_experiments
from lm_human_preferences import lm_tasks
from lm_human_preferences.language import trained_models, lm_datasets
from lm_human_preferences.params import RunHParams, TaskHParams
from lm_human_preferences.policy import Policy
from lm_human_preferences.utils import launch


def _get_policy(policy_cktp: Path, task: TaskHParams):
    run_params = RunHParams()
    m = trained_models.TrainedModel(policy_cktp, run_hparams=run_params)
    encoder = m.encoding.get_encoder()

    return Policy(
        m, encoder,
        embed_queries=lm_tasks.query_formatter(task, encoder),
        temperature=task.policy.temperature)


@torch.no_grad()
def policy_respond(context: str, policy_cktp: str, experiment: str, **kwargs):
    """
    Given context return response
    """
    if not context or not experiment:
        raise ValueError("Please provide some context text and experiment name for example \'sentiment\'.")
    policy_cktp = Path(policy_cktp)
    assert policy_cktp.is_file(), "Policy checkpoint does not exist."

    trial = get_experiments()[experiment]
    task, _ = launch.params(trial, TaskHParams, kwargs)

    policy = _get_policy(policy_cktp, task)
    policy.eval()
    encoder = policy.encoder

    start_token = encoder.encode(task.start_text) if task.start_text else None
    end_token = encoder.encode(task.end_text) if task.end_text else None
    tokens = lm_datasets.prepare_token(context, encoder, start_token, end_token, task.query_length)
    query = torch.as_tensor(tokens, dtype=torch.int32, device=policy.device)
    response = policy.respond(query, length=task.response_length)['responses']
    print(f"query is: {query}")
    print(f"response is: {response}")


if __name__ == '__main__':
    launch.main(dict(
        #sample=sample_policy,
        respond=policy_respond
    ))

"""
context=hello world
policy_ckpt=/path/to/checkpoint
./sample.py sample --context $context --policy_ckpt $policy_ckpt --experiment sentiment
"""
