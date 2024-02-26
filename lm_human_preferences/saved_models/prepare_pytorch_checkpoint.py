import os
import re

import numpy as np
import tensorflow as tf
import torch

from lm_human_preferences.language.gpt import ModelParams, GPT

""" Prepare PyTorch model checkpoint from downloaded OpenAI's Tensorflow checkpoint
from 
https://openaipublic.blob.core.windows.net/gpt-2/models/124M/model.ckpt.data-00000-of-00001
https://openaipublic.blob.core.windows.net/gpt-2/models/124M/model.ckpt.index
https://openaipublic.blob.core.windows.net/gpt-2/models/124M/model.ckpt.meta
 
ref: https://medium.com/huggingface/from-tensorflow-to-pytorch-265f40ef2a28
"""

tf_path = os.path.abspath('./model_124M/model.ckpt')  # Path to our TensorFlow checkpoint
init_vars = tf.train.list_variables(tf_path)  # a list of (name, shape) tuples
tf_vars: list[tuple[str, np.ndarray]] = []
for name, shape in init_vars:
    array = tf.train.load_variable(tf_path, name)
    tf_vars.append((name, array.squeeze()))

hparams = ModelParams()
model = GPT(hparams)
sd = model.state_dict()
for name, array in tf_vars:
    name_segs = ['transformer'] + name[len('model/'):].split('/')
    m = re.fullmatch(r'([A-Za-z]+)(\d+)', name_segs[1])
    if m:
        name_segs[1] = m.group(1)  # m.group(1) is 'h'
        name_segs.insert(2, m.group(2))  # m.group(2) is the layer number like '0'
    if name_segs[-1] in 'gw':
        name_segs[-1] = 'weight'
    elif name_segs[-1] == 'b':
        name_segs[-1] = 'bias'
    p_name = '.'.join(name_segs)  # variable name in Pytorch model except the embeddings
    with torch.no_grad():
        # embeddings are different from the transformer layers
        if 'wpe' in p_name or 'wte' in p_name:
            p_name = '.'.join((p_name, 'weight'))
            assert tuple(sd[p_name].shape) == array.shape
            sd[p_name].copy_(torch.as_tensor(array))
        else:
            # tf.layer.dense whose kernel is the transposed of PyTorch's nn.Linear weights.
            # And transpose on 1-d numpy has no effect.
            assert tuple(sd[p_name].shape[::-1]) == array.shape
            sd[p_name].copy_(torch.as_tensor(array.T))  # transpose!

# save state_dict to checkpoint
ckpt = {'model': model.state_dict()}
torch.save(ckpt, os.path.join(os.path.dirname(__file__), '124M_ckpt.pt'))  # name matches params.RunHParams.ckpt


"""
Tensorflow weights are like:

('model/ln_f/b', [768]),
('model/ln_f/g', [768]),
('model/wpe', [1024, 768]),
('model/wte', [50257, 768])]

('model/h0/attn/c_attn/b', [2304]),
('model/h0/attn/c_attn/w', [1, 768, 2304]),
('model/h0/attn/c_proj/b', [768]),
('model/h0/attn/c_proj/w', [1, 768, 768]),
('model/h0/ln_1/b', [768]),
('model/h0/ln_1/g', [768]),
('model/h0/ln_2/b', [768]),
('model/h0/ln_2/g', [768]),
('model/h0/mlp/c_fc/b', [3072]),
('model/h0/mlp/c_fc/w', [1, 768, 3072]),
('model/h0/mlp/c_proj/b', [768]),
('model/h0/mlp/c_proj/w', [1, 3072, 768]),

Pytorch weights are like:

('transformer.wte.weight', ([50257, 768])),
('transformer.wpe.weight', ([1024, 768])),
('transformer.ln_f.weight', ([768])),
('transformer.ln_f.bias', ([768])),

('transformer.h.0.attn.c_attn.bias', ([2304])),
('transformer.h.0.attn.c_attn.weight', ([2304, 768])),
('transformer.h.0.attn.c_proj.bias', ([768])),
('transformer.h.0.attn.c_proj.weight', ([768, 768])),
('transformer.h.0.ln_1.weight', ([768])),
('transformer.h.0.ln_1.bias', ([768])),
('transformer.h.0.ln_2.weight', ([768])),
('transformer.h.0.ln_2.bias', ([768])),
('transformer.h.0.mlp.c_fc.weight', ([3072, 768])),
('transformer.h.0.mlp.c_fc.bias', ([3072])),
('transformer.h.0.mlp.c_proj.weight', ([768, 3072])),
('transformer.h.0.mlp.c_proj.bias', ([768])),
"""
