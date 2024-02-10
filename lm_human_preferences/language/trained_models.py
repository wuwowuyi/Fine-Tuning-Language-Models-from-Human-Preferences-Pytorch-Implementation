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


class TrainedModel:
    def __init__(self, name, *, run_hparams: RunHParams):
        self.name = name  # for example, 124M
        self.savedir: Path = Path(run_hparams.save_dir)  # savedir cannot be None. we don's save to gcs.
        self.ckpt = run_hparams.ckpt  # checkpoint
        self.device = run_hparams.device
        if name == 'test':
            self.encoding = encodings.Test
        else:
            self.encoding = encodings.Main

    def _checkpoint(self, model_for: str):
        if self.name == 'test':
            return None
        ckpt_path = self.savedir / model_for / self.ckpt
        if ckpt_path.is_file():
            print(f"Load checkpoint from {ckpt_path}")
            return torch.load(ckpt_path, map_location=self.device)

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
        :return: a language model instance
        """
        checkpoint = self._checkpoint(model_for)
        model_args: gpt.ModelParams = self._hparams(model_for)
        if checkpoint:  # a checkpoint exists, load model from it.
            checkpoint_model_args = checkpoint['model_args']  # dict
            # force these config attributes to be equal
            # the rest of the attributes (e.g. dropout) can stay as desired from command line
            for k in ['n_layer', 'n_head', 'n_embd', 'n_ctx', 'bias', 'n_vocab']:
                setattr(model_args, k, checkpoint_model_args[k])

            state_dict = checkpoint['model']
            # fix the keys of the state dictionary :(
            unwanted_prefix = '_orig_mod.'
            for k, v in list(state_dict.items()):
                if k.startswith(unwanted_prefix):
                    state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)

            model = gpt.GPT(model_args)
            model.load_state_dict(state_dict)
        else:
            print(f"Initializing from OpenAI GPT-2 weights: {self.name}")
            # initialize from OpenAI GPT-2 weights
            override_args = dict(embd_pdrop=model_args.embd_pdrop,
                                 attn_pdrop=model_args.attn_pdrop,
                                 resid_pdrop=model_args.resid_pdrop,
                                 head_pdrop=model_args.head_pdrop)
            model = gpt.GPT.from_pretrained(GPT2_Model[self.name], override_args)  # make sure hparams matches pretrained hparams
            for k in ['n_layer', 'n_head', 'n_embd', 'n_ctx', 'bias', 'n_vocab']:
                setattr(model_args, k, getattr(model.config, k))

            # overwrite default init of the head layer for policy/reward
            if model_for == 'policy':
                torch.nn.init.zeros_(model.hp_head.weight)  # TODO: to review. zero initial value?
            else:
                torch.nn.init.normal_(model.hp_head.weight, std=1 / np.sqrt(model_args.n_embd + 1))
        # todo: save mode_args here?
        return model, model_args


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
