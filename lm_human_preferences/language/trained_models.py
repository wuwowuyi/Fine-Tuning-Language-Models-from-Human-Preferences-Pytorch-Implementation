from pathlib import Path

import numpy as np
import torch

from lm_human_preferences.language import encodings, gpt
from lm_human_preferences.language.gpt import GPT

GPT2_Model = {  # mapping to match hugging face's model names
    '124M': 'gpt2',
    '350M': 'gpt2-medium',
    '774M': 'gpt2-large',
    '1558M': 'gpt2-xl'
}


class TrainedModel():
    def __init__(self, name, *, run_hparams):
        self.name = name  # for example, 124M
        self.savedir: Path = Path(run_hparams.save_dir)  # savedir cannot be None. we don's save to gcs.
        self.ckpt = run_hparams.ckpt  # checkpoint
        self.device = run_hparams.device

        if name == 'test':
            self.encoding = encodings.Test
        else:
            self.encoding = encodings.Main
        self._hparams = None

    def checkpoint(self, model_type: str):
        if self.name == 'test':
            return None
        ckpt_path = self.savedir / model_type / self.ckpt
        if ckpt_path.is_file():
            print(f"Load checkpoint from {ckpt_path}")
            return torch.load(ckpt_path, map_location=self.device)

    def hparams(self):
        if self._hparams is None:
            if self.name == 'test':
                hparams = test_hparams()
            else:
                hparams = gpt.HParams()  # default hyperparams
                f = self.savedir / 'hparams.json'  # override from a json file
                if f.is_file():
                    hparams.override_from_json_file(f)
            self._hparams = hparams
        return self._hparams

    def init_model(self, model_type: str):
        """
        :param model_type: 'policy', 'reward', etc.
        :return: a language model instance
        """
        checkpoint = self.checkpoint(model_type)
        config = self.hparams()
        if checkpoint:  # a checkpoint exists, load model from it.
            state_dict = checkpoint['model']
            # fix the keys of the state dictionary :(
            unwanted_prefix = '_orig_mod.'
            for k, v in list(state_dict.items()):
                if k.startswith(unwanted_prefix):
                    state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)
            model = GPT(config)
            model.load_state_dict(state_dict)
        else:
            print(f"Initializing from OpenAI GPT-2 weights: {self.name}")
            # initialize from OpenAI GPT-2 weights
            model = GPT.from_pretrained(GPT2_Model[self.name])  # make sure hparams matches pretrained hparams
            # overwrite default init of the head layer for policy/reward
            if model_type == 'policy':
                torch.nn.init.zeros_(model.hp_head.weight)  # TODO: zero initial value?
            else:
                torch.nn.init.normal_(model.hp_head.weight, std=1 / np.sqrt(config.n_embd + 1))
        model.to(self.device)
        return model


def load_hparams(file):
    hparams = gpt.HParams()
    hparams.override_from_json_file(file)
    return hparams


def test_hparams():
    hparams = gpt.HParams()
    hparams.override_from_dict(dict(
        n_vocab=27,  # Corresponds to random encoding length
        n_ctx=8,
        n_layer=2,
        n_embd=7,
        n_head=1,
    ))
    return hparams
