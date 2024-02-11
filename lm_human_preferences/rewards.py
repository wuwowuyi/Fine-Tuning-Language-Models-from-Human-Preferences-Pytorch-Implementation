"""Synthetic scores."""

import torch
from torch import nn


# TODO: sort out device and gradient!

class RewardModel(nn.Module):
    def __init__(
            self,
            trained_model,
            encoder
    ):
        super().__init__()
        self.trained_model = trained_model
        self.device = self.trained_model.device
        self.encoder = encoder
        self.padding_token = self.encoder.padding_token

        # also use a gpt-2 model
        self.lm_model, self.lm_params = self.trained_model.init_model('reward')  # pre-trained language model
        self.reward_gain = nn.Parameter(torch.ones(1, device=self.device))
        self.reward_bias = nn.Parameter(torch.zeros(1, device=self.device))

    def get_encoder(self):
        return self.encoder

    def forward(self, tokens):
        lm_output = self.lm_model(tokens, padding_token=self.padding_token)
        reward = lm_output['hp'][:, -1]  # shape=(b,)
        return self.reward_gain * reward + self.reward_bias

    def reset_reward_scale(self):
        self.reward_gain.copy_(torch.tensor(1))
        self.reward_bias.copy_(torch.zeros(1))

    def set_reward_norm(self, *, old_mean, old_std, new_mean, new_std):
        """Given old_mean+-old_std of reward_model, change gain and bias to get N(new_mean,new_std)."""
        old_gain, old_bias = self.reward_gain, self.reward_bias
        assert old_gain == 1 and old_bias == 0,\
            f'set_reward_norm expects gain = 1 and bias = 0, not {old_gain}, {old_bias}'
        gain = new_std / old_std
        bias = new_mean - gain * old_mean
        self.reward_gain.copy_(torch.as_tensor(gain))
        self.reward_bias.copy_(torch.as_tensor(bias))

    def get_rewards(self, queries, responses):
        tokens = torch.concat((queries, responses), dim=1)
        return self(tokens)

    def configure_optimizers(self, hparams):
        device_type = 'cuda' if 'cuda' in self.device else 'cpu'
        return self.lm_model.configure_optimizers(
            hparams.weight_decay, hparams.lr, hparams.betas, device_type,
            {'reward_gain': self.reward_gain, 'reward_bias': self.reward_bias}
        )


# class TrainedRewardModel():
#     def __init__(self, train_dir, encoding, *, scope='reward_model', comm=MPI.COMM_WORLD):
#         self.train_dir = train_dir
#         self.comm = comm
#
#         self.encoding = encoding
#         encoder = encoding.get_encoder()
#         if train_dir != 'test':
#             self.hparams = trained_models.load_hparams(os.path.join(train_dir, 'hparams.json'))
#             assert self.hparams.n_vocab == encoding.n_vocab, f'{self.hparams.n_vocab} != {encoding.n_vocab}'
#         else:
#             self.hparams = trained_models.test_hparams()
#
#         self.padding_token = encoder.padding_token
#
#         self.encoder = encoder
#
#         self.scope = scope
#         self.model = model.Model(hparams=self.hparams, scope=f'{scope}/model', scalar_heads=['reward'])
#
#     def _build(self, X):
#         results = self.model(X=X, padding_token=self.padding_token)
#         reward = results['reward'][:, -1]
#         with tf.compat.v1.variable_scope(f'{self.scope}/reward_norm'):
#             self.reward_gain = tf.compat.v1.get_variable('gain', shape=(), initializer=tf.compat.v1.constant_initializer(1))
#             self.reward_bias = tf.compat.v1.get_variable('bias', shape=(), initializer=tf.compat.v1.constant_initializer(0))
#         reward = self.reward_gain * reward + self.reward_bias
#         self._set_initializers()
#         return reward
#
#     def ensure_built(self):
#         if self.model.built:
#             return
#         with tf.compat.v1.name_scope('dummy'):
#             self._build(X=tf.zeros([0,0], dtype=tf.int32))
#
#     def _set_initializers(self):
#         """Change initializers to load a model from a tensorflow checkpoint."""
#         if self.comm.Get_rank() > 0 or self.train_dir == 'test':
#             return
#
#         assert self.model.built
#         checkpoint_scope = 'reward_model'
#
#         with tf.init_scope():
#             # Initialize!
#             params = {v.op.name: v for v in self.get_params()}
#             checkpoint = tf.train.latest_checkpoint(os.path.join(self.train_dir, 'checkpoints/'))
#             available = tf.train.list_variables(checkpoint)
#             unchanged = {}
#
#             for name, shape in available:
#                 if not name.startswith(checkpoint_scope + '/'):
#                     # print('skipping', name)
#                     continue
#                 if name.endswith('adam') or name.endswith('adam_1'):
#                     # print('skipping', name)
#                     continue
#                 print('setting', name)
#                 var = params[self.scope + name[len(checkpoint_scope):]]
#                 assert var.shape == shape, 'Shape mismatch: %s.shape = %s != %s' % (var.op.name, var.shape, shape)
#                 unchanged[name] = var
#             tf.compat.v1.train.init_from_checkpoint(checkpoint, unchanged)
#
#     def get_params(self):
#         return self.model.get_params() + [self.reward_gain, self.reward_bias]
#
#     def score_fn(self, queries, responses):
#         tokens = tf.concat([queries, responses], axis=1)
#         return self._build(tokens)
