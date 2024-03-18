from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import torch

from lm_human_preferences.utils import core_torch as utils, hyperparams


@dataclass
class LabelHParams(hyperparams.HParams):
    type: str = None
    num_train: int = None
    source: str = None


@dataclass
class RunHParams(hyperparams.HParams):
    seed: Optional[int] = None
    log_interval: int = 10
    save_interval: int = 50

    # We always save and load from a local dir.
    # save_dir is for a particular job
    # The checkpoint of language model/reward/policy for initialization is under save_dir.parent
    save_dir: Optional[Path] = None

    device: str = 'cuda' if torch.cuda.is_available() else 'cpu'  # 'cpu', 'cuda', 'cuda:0', 'cuda:1' etc.
    ckpt: str = '124M_ckpt.pt'  # language model checkpoint
    output_ckpt: str = 'output_ckpt.pt'  # trained reward/policy checkpoint

    # init: init both policy and reward from the downloaded LM model.
    # policy: reward model is trained. init policy from the downloaded LM, reward from saved checkpoint
    # resume: init both policy and reward from saved checkpoints
    train_stage: str = 'init'

    # wandb logging
    wandb_log: bool = False
    wandb_project: str = 'lm_human_preference'


@dataclass
class PolicyHParams(hyperparams.HParams):
    temperature: float = 1.0  # lower this number to improve sample quality.
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


@dataclass
class TrainRewardParams(hyperparams.HParams):
    run: RunHParams = field(default_factory=RunHParams)

    task: TaskHParams = field(default_factory=TaskHParams)
    labels: LabelHParams = field(default_factory=LabelHParams)

    batch_size: int = 40  # micro_batch_size = batch_size / gradient_accumulation_steps
    gradient_accumulation_steps: int = 1  # increase to avoid OutOfMemory Error
    lr: float = 5e-5
    grad_clip = 1.0

    rollout_batch_size: int = 64
    normalize_samples: int = 0  # Samples used to estimate reward mean and std
    debug_normalize: int = 0  # Samples used to check that normalization worked
    # Whether, before training, to normalize the rewards on the policy to the scales on the training buffer.
    # (For comparisons, just use mean 0, var 1.)
    normalize_before: bool = False
    # Whether, after training, to normalize the rewards on the ref policy to mean 0, var 1
    # (so the KL coefficient always has the same meaning).
    normalize_after: bool = False

    def validate(self, *, prefix=''):
        super().validate(prefix=prefix)
        utils.exact_div(self.labels.num_train, self.batch_size)


@dataclass
class AdaptiveKLParams(hyperparams.HParams):
    target: float = None
    horizon: int = 10000  # in episodes


@dataclass
class RewardHParams(hyperparams.HParams):
    kl_coef: float = 0.2
    adaptive_kl: Optional[AdaptiveKLParams] = None

    trained_model: Optional[str] = None

    train_new_model: Optional[TrainRewardParams] = None

    def validate(self, *, prefix=''):
        super().validate(prefix=prefix)
        assert self.trained_model is None or self.train_new_model is None, 'Cannot use trained_model and train new model'
        assert self.trained_model is not None or self.train_new_model is not None, 'Need either trained_model or to train a new model'


@dataclass
class PpoHParams(hyperparams.HParams):
    total_episodes: int = 2000000
    batch_size: int = 64
    nminibatches: int = 8  # increase to 8 since we train on a single GPU
    noptepochs: int = 4  # each batch is trained this number of times

    lr: float = 5e-6
    vf_coef: float = .1
    cliprange: float = .2
    cliprange_value: float = .2
    gamma: float = 1
    lam: float = 0.95
    whiten_rewards: bool = True


@dataclass
class TrainPolicyParams(hyperparams.HParams):
    run: RunHParams = field(default_factory=RunHParams)

    task: TaskHParams = field(default_factory=TaskHParams)
    rewards: RewardHParams = field(default_factory=RewardHParams)
    ppo: PpoHParams = field(default_factory=PpoHParams)

    def validate(self, *, prefix=''):
        super().validate(prefix=prefix)
        # NOTE: must additionally divide by # ranks
        minibatch_size = utils.exact_div(self.ppo.batch_size, self.ppo.nminibatches)
        if self.ppo.whiten_rewards:
            assert minibatch_size >= 8, \
                f"Minibatch size {minibatch_size} is insufficient for whitening in PPOTrainer.loss"
