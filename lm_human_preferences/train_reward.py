#!/usr/bin/env python3

import json

import numpy as np
import torch
import wandb

from lm_human_preferences import label_types, lm_tasks, rewards
from lm_human_preferences.label_types import LabelType
from lm_human_preferences.language import trained_models
from lm_human_preferences.params import TrainRewardParams
from lm_human_preferences.policy import Policy
from lm_human_preferences.utils import azure, hyperparams
from lm_human_preferences.utils import core_torch as utils


def download_labels(source: str, label_type: LabelType,
                    question_schemas: dict[str, utils.Schema], total_labels: int):
    schemas = {**question_schemas, **label_type.label_schemas()}

    """
    if self.is_root:
        with tf.device('cpu:0'):
            self._enqueue_phs = {
                name: tf.placeholder(name=name, dtype=schema.dtype, shape=(None,) + schema.shape)
                for name, schema in self.schemas.items()
            }
            self._enqueue_answers = self.answer_queue.enqueue_many(self._enqueue_phs)
    else:
        self._enqueue_phs = None
        self._enqueue_answers = None
    """

    if source != 'test':
        with open(azure.download_file_cached(source)) as f:
            results = json.load(f)
            print('Num labels found in source:', len(results))
    else:
        results = [
            {
                name: np.zeros(schema.shape, dtype=schema.dtype.as_numpy_dtype)
                for name, schema in schemas.items()
            }
            for _ in range(50)
        ]

    assert len(results) >= total_labels
    # results is a list of items in schemas' format. eg,
    # [{'query':..., 'sample0':..., 'sample1':..., 'sample2':,, 'sample3':.., 'best':..},...]
    results = results[:total_labels]
    return {k: torch.as_tensor([a[k] for a in results]) for k in schemas.keys()}


class RewardModelTrainer:
    def __init__(self, *, reward_model, policy, query_sampler, hparams: TrainRewardParams):
        self.reward_model = reward_model
        self.policy = policy
        self.hparams = hparams

        self.label_type = label_types.get(hparams.labels.type)  # e.g., best_of_4
        self.question_schemas = self.label_type.question_schemas(
            query_length=hparams.task.query_length,
            response_length=hparams.task.response_length,
        )
        data_schemas = {
            **self.question_schemas,
            **self.label_type.label_schemas(),
        }
        self.train_buffer = utils.SampleBuffer(capacity=hparams.labels.num_train, schemas=data_schemas)

        if self.hparams.normalize_before or self.hparams.normalize_after:

            def target_mean_std():
                """Returns the means and variances to target for each reward model"""
                # Should be the same on all ranks because the train_buf should be the same
                scales = self.label_type.target_scales(self.train_buffer.data())
                if scales is None:
                    return torch.zeros([]), torch.ones([])
                else:
                    var, mean = torch.var_mean(scales, dim=0, correction=0)
                    return mean, torch.sqrt(var)
            self.target_mean_std = target_mean_std

            @torch.no_grad()
            def stats(query_responses):
                rewards = torch.cat([self.reward_model.get_rewards(qs, rs) for qs, rs in query_responses], dim=0)
                assert len(rewards.shape) == 1, f'{rewards.shape}'
                means, sqr_means = rewards.mean(), rewards.square().mean()
                stds = torch.sqrt(sqr_means - means ** 2)  # Var(x) = E(x^2) - (E[x]^2)
                return means, stds
            self.stats = stats

            def log_stats_after_normalize(stats):
                means, stds = stats
                print(f'after normalize: {means} +- {stds}')
            self.log_stats_after_normalize = log_stats_after_normalize

            def reset_reward_scales():
                self.reward_model.reset_reward_scale()
            self.reset_reward_scales = reset_reward_scales

            def set_reward_norms(mean, std, new_mean, new_std):
                print(f'targets: {new_mean} +- {new_std}')
                print(f'before normalize: {mean} +- {std}')
                mean, std = mean.cpu(), std.cpu()
                new_mean, new_std = new_mean.cpu(), new_std.cpu()
                assert np.isfinite((mean.numpy(), std.numpy(), new_mean.numpy(), new_std.numpy())).all()
                self.reward_model.set_reward_norm(old_mean=mean, old_std=std, new_mean=new_mean, new_std=new_std)
            self.set_reward_norms = set_reward_norms

        if self.hparams.normalize_before or self.hparams.normalize_after:

            def sample_policy_batch():
                queries = query_sampler()
                responses = policy.respond(
                    queries=queries, length=hparams.task.response_length)['responses']
                return queries, responses

            def sample_policy_responses(n_samples):
                n_batches = utils.ceil_div(n_samples, hparams.rollout_batch_size)
                return [sample_policy_batch() for _ in range(n_batches)]
            self.sample_policy_responses = sample_policy_responses

        def add_to_buffer(labels):
            return self.train_buffer.add(**labels)
        self.add_to_buffer = add_to_buffer

    def normalize(self, sample_fn, target_means, target_stds):
        if not self.hparams.normalize_samples:
            return

        self.reset_reward_scales()
        query_responses = sample_fn(self.hparams.normalize_samples)
        means, stds = self.stats(query_responses)
        self.set_reward_norms(means, stds, target_means, target_stds)

        if self.hparams.debug_normalize:
            query_responses = sample_fn(self.hparams.debug_normalize)
            stats = self.stats(query_responses)
            self.log_stats_after_normalize(stats)

    def train(self):
        labels = download_labels(
            self.hparams.labels.source,
            label_type=self.label_type,
            question_schemas=self.question_schemas,
            total_labels=self.hparams.labels.num_train
        )

        self.add_to_buffer(labels)

        if self.hparams.normalize_before:
            target_mean, target_std = self.target_mean_std()
            self.normalize(self.sample_policy_responses, target_mean, target_std)

        optimizer = self.reward_model.configure_optimizers(self.hparams)

        train_indices = torch.randperm(self.hparams.labels.num_train)
        for index in range(self.hparams.labels.num_train // self.hparams.batch_size):
            start_index = index * self.hparams.batch_size
            end_index = start_index + self.hparams.batch_size
            indices = train_indices[start_index:end_index]

            minibatch = self.train_buffer.read(indices)
            for k, v in minibatch.items():
                minibatch[k] = torch.as_tensor(v, dtype=torch.int32, device=self.hparams.run.device)

            stats = self.label_type.loss(reward_model=self.reward_model.get_rewards, labels=minibatch)
            loss = stats['loss'] / self.hparams.gradient_accumulation_steps
            loss.backward()

            if index % self.hparams.run.log_interval == 0:
                if self.hparams.wandb_log:
                    wandb.log({
                        "iter": index,
                        "train/loss": loss,
                        #"val/loss": losses['val'],
                        #"lr": lr,
                    })
                print(f"iter {index}: loss {loss:.4f}")

            if (index + 1) % self.hparams.gradient_accumulation_steps == 0:
                # clip the gradient
                if self.hparams.grad_clip != 0.0:
                    torch.nn.utils.clip_grad_norm_(self.reward_model.parameters(), self.hparams.grad_clip)
                optimizer.step()
                # flush the gradients as soon as we can, no need for this memory anymore
                optimizer.zero_grad(set_to_none=True)

                # schedule learning rate
                lr = (1 - start_index / self.hparams.labels.num_train) * self.hparams.lr
                for param_group in optimizer.param_groups:
                    param_group['lr'] = lr

            # todo: save checkpoint

        if self.hparams.normalize_after:
            target_mean, target_std = np.zeros([]), np.ones([])
            self.normalize(self.sample_policy_responses, target_mean, target_std)


def train(hparams: TrainRewardParams):
    hyperparams.dump(hparams)  # output hparams to out (default to stdout)

    m = trained_models.TrainedModel(hparams.task.policy.initial_model, run_hparams=hparams.run)
    encoder = m.encoding.get_encoder()

    # only used as a language model for sampling responses given context
    ref_policy = Policy(
        m, encoder,
        embed_queries=lm_tasks.query_formatter(hparams.task, encoder),
        temperature=hparams.task.policy.temperature)

    reward_model = rewards.RewardModel(m, encoder)

    query_sampler = lm_tasks.make_query_sampler(
        hparams=hparams.task, encoder=encoder, batch_size=hparams.rollout_batch_size, device=hparams.run.device
    )

    reward_trainer = RewardModelTrainer(
        reward_model=reward_model,
        policy=ref_policy,
        query_sampler=query_sampler,
        hparams=hparams,
    )

    #
    # save_dir = hparams.run.save_dir
    #
    # print(f"Will save to {save_dir}")
    # saver = tf.compat.v1.train.Saver(max_to_keep=20, save_relative_paths=True)
    # checkpoint_dir = os.path.join(save_dir, 'reward_model/checkpoints/model.ckpt')
    #
    # if not save_dir.startswith('gs://'):
    #     os.makedirs(os.path.join(save_dir, 'reward_model'), exist_ok=True)
    # with tf.io.gfile.GFile(os.path.join(save_dir, 'train_reward_hparams.json'), 'w') as f:
    #     json.dump(hparams.to_nested_dict(), f, indent=2)
    # with tf.io.gfile.GFile(os.path.join(save_dir, 'reward_model', 'hparams.json'), 'w') as f:
    #     json.dump(reward_model.hparams.to_nested_dict(), f, indent=2)
    # with tf.io.gfile.GFile(os.path.join(save_dir, 'reward_model', 'encoding'), 'w') as f:
    #     json.dump(reward_model.trained_model.encoding.name, f, indent=2)

    reward_trainer.train()
