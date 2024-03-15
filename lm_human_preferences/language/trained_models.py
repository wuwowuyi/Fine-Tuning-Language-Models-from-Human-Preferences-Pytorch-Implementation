from pathlib import Path

import numpy as np
import torch

from lm_human_preferences.language import encodings, gpt
from lm_human_preferences.params import RunHParams

GPT2_Model = {  # mapping to match hugging face's model names
    '124M': 'gpt2',
    '350M': 'gpt2-medium',
    '774M': 'gpt2-large',
    '1558M': 'gpt2-xl'
}


# This class is used as a utility class to initialize reward and policy.
class TrainedModel:
    def __init__(self, name, *, run_hparams: RunHParams):
        self.name = name  # for example, 124M
        self.savedir: Path = run_hparams.save_dir  # save dir for a particular run job.
        self.lm_ckpt = run_hparams.ckpt  # pretrained language model checkpoint
        self.output_ckpt = run_hparams.output_ckpt

        self.device = run_hparams.device
        self.train_stage = run_hparams.train_stage

        if name == 'test':
            self.encoding = encodings.Test
        else:
            self.encoding = encodings.Main

    def get_ckpt_filename(self, model_for: str, use_parent=False) -> Path:
        """Return filename for model checkpoint. """
        p = self.savedir.parent / model_for if use_parent else self.savedir / model_for
        if not p.exists():
            p.mkdir()
        return p / self.output_ckpt

    def _checkpoint(self, model_for: str):
        """
        Load checkpoint for model.
        All checkpoints are stored under `self.savedir`.

        - If `train_stage` is "init", initialize both reward and policy from the pretrained language model
        - If `train_stage` is "policy", init policy from the pretrained language model,
            and reward from the checkpoint saved during training reward.
        - Otherwise load both reward and policy from saved checkpoint.

        """
        def _load_ckpt(ckpt: Path):
            if ckpt.is_file():
                print(f"Load checkpoint from {ckpt}")
                return torch.load(ckpt, map_location=self.device)
            else:
                raise ValueError(f"checkpoint does not exist: {ckpt}")

        if self.train_stage == 'init' or (self.train_stage == 'policy' and model_for == 'policy'):
            # init from the downloaded pre-trained language model
            return _load_ckpt(self.savedir.parent / self.lm_ckpt)
        else:  # init from saved checkpoint
            return _load_ckpt(self.get_ckpt_filename(model_for, True))

    def _hparams(self, model_for: str):
        if self.name == 'test':
            hparams = test_hparams()
        else:
            hparams = gpt.ModelParams()  # default hyperparams
            f = self.savedir / model_for / 'hparams.json'  # override from a json file
            if f.is_file():
                hparams.override_from_json_file(f)
        return hparams

    def init_model(self, model_for: str):
        """
        :param model_for: 'policy', 'reward', etc.
        """
        checkpoint = self._checkpoint(model_for)  # NOTE: we always save checkpoint as a dict
        assert checkpoint is not None

        model_args: gpt.ModelParams = self._hparams(model_for)
        if 'model_args' in checkpoint:
            checkpoint_model_args = checkpoint.pop('model_args')  # dict
            # force these config attributes to be equal
            # the rest of the attributes (e.g. dropout) can stay as desired from command line
            for k in ['n_layer', 'n_head', 'n_embd', 'n_ctx', 'bias', 'n_vocab']:
                setattr(model_args, k, checkpoint_model_args[k])

        state_dict = checkpoint.pop('model')
        # fix the keys of the state dictionary :(
        unwanted_prefix = '_orig_mod.'
        for k, v in list(state_dict.items()):
            if k.startswith(unwanted_prefix):
                state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)

        model = gpt.GPT(model_args)
        model.load_state_dict(state_dict)

        # overwrite default init of the head layer for policy/reward
        if self.train_stage == 'init':
            if model_for == 'policy':
                torch.nn.init.zeros_(model.hp_head.weight)  # TODO: to review. zero initial value?
            else:  # reward
                torch.nn.init.normal_(model.hp_head.weight, std=1 / np.sqrt(model_args.n_embd + 1))
        elif self.train_stage == 'policy' and model_for == 'policy':
            torch.nn.init.zeros_(model.hp_head.weight)  # TODO: to review. zero initial value?

        model.to(self.device)
        return model, model_args, checkpoint


def load_hparams(file):
    hparams = gpt.ModelParams()
    hparams.override_from_json_file(file)
    return hparams


def test_hparams():
    hparams = gpt.ModelParams()
    hparams.override_from_dict(dict(
        n_vocab=27,  # Corresponds to random encoding length
        n_ctx=8,
        n_layer=2,
        n_embd=7,
        n_head=1,
    ))
    return hparams
