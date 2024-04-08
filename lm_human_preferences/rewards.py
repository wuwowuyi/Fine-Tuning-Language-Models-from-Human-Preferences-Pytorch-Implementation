"""Synthetic scores."""
import os

import torch
from torch import nn

from lm_human_preferences.language.trained_models import TrainedModel
from lm_human_preferences.params import TrainRewardParams


class RewardModel(nn.Module):
    def __init__(
            self,
            trained_model: TrainedModel,
            encoder
    ):
        super().__init__()
        self.trained_model = trained_model
        self.device = self.trained_model.device
        self.encoder = encoder
        self.padding_token = self.encoder.padding_token

        self.lm_model, self.lm_params, ckpt = self.trained_model.init_model('reward')
        self.model = self.lm_model.module if self.trained_model.ddp else self.lm_model

        self.reward_gain = ckpt.pop('gain') if 'gain' in ckpt else nn.Parameter(torch.ones((), device=self.device))
        self.reward_bias = ckpt.pop('bias') if 'bias' in ckpt else nn.Parameter(torch.zeros((), device=self.device))

    def forward(self, tokens):
        """Only care the reward for the entire response, not per step.
        Since the reward is trained on scores for the entire response.
         """
        lm_output = self.lm_model(tokens, padding_token=self.padding_token)
        reward = lm_output['hp'][:, -1]  # shape=(b,) where b=tokens.shape[0]
        return self.reward_gain * reward + self.reward_bias  # shape=(b, ) where b=tokens.shape[0]

    @torch.no_grad()
    def reset_reward_scale(self):
        self.reward_gain.copy_(torch.ones(()))
        self.reward_bias.copy_(torch.zeros(()))

    @torch.no_grad()
    def set_reward_norm(self, *, old_mean, old_std, new_mean, new_std):
        """Given old_mean+-old_std of reward_model, change gain and bias to get N(new_mean,new_std)."""
        old_gain, old_bias = self.reward_gain, self.reward_bias
        assert old_gain == 1 and old_bias == 0,\
            f'set_reward_norm expects gain = 1 and bias = 0, not {old_gain}, {old_bias}'
        gain = new_std / old_std
        bias = new_mean - gain * old_mean
        self.reward_gain.copy_(gain)
        self.reward_bias.copy_(bias)

    def get_rewards(self, queries, responses):
        tokens = torch.concat((queries, responses), dim=1)
        return self(tokens)

    def configure_optimizers(self, hparams: TrainRewardParams):
        device_type = 'cuda' if 'cuda' in self.device else self.device
        return self.model.configure_optimizers(
            hparams.lr, device_type, extra_params={'reward_gain': self.reward_gain, 'reward_bias': self.reward_bias}
        )

    def save(self):
        ckpt = {
            'model': self.model.state_dict(),
            'gain': self.reward_gain,
            'bias': self.reward_bias
        }
        f = self.trained_model.get_ckpt_filename('reward')
        torch.save(ckpt, f)
        print(f'Reward model has been saved to {f}')

    def set_grad_sync(self, sync: bool):
        self.lm_model.require_backward_grad_sync = sync

