#!/usr/bin/env python3

import json
from contextlib import nullcontext

import numpy as np
import torch
import tqdm
import wandb
from torch.distributed import destroy_process_group

from lm_human_preferences import label_types, lm_tasks, rewards, params
from lm_human_preferences.language import trained_models
from lm_human_preferences.params import TrainRewardParams
from lm_human_preferences.policy import Policy
from lm_human_preferences.utils import azure, hyperparams
from lm_human_preferences.utils import core_torch as utils


def download_labels(source: str, schemas: dict[str, utils.Schema], total_labels: int, labels_dir='/tmp/azure-cache'):

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
        with open(azure.download_file_cached(source, labels_dir)) as f:
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
    return {k: torch.as_tensor([a[k] for a in results], dtype=schema.dtype, device='cpu')
            for k, schema in schemas.items()}


class RewardModelTrainer:
    def __init__(self, *, reward_model, policy, query_sampler, hparams: TrainRewardParams):
        self.reward_model = reward_model
        self.policy = policy
        self.hparams = hparams

        self.label_type = label_types.get(hparams.labels.type)  # e.g., best_of_4
        question_schemas = self.label_type.question_schemas(
            query_length=hparams.task.query_length,
            response_length=hparams.task.response_length,
        )
        self.data_schemas = {
            **question_schemas,
            **self.label_type.label_schemas(),
        }
        self.train_buffer = utils.SampleBuffer(capacity=hparams.labels.num_train, schemas=self.data_schemas)

        self.ptdtype: torch.dtype = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float16
        self.run_ctx = nullcontext() if self.hparams.run.device == 'cpu' \
            else torch.amp.autocast(device_type='cuda', dtype=self.ptdtype)

        if self.hparams.normalize_before or self.hparams.normalize_after:

            def target_mean_std():
                """Returns the means and variances to target for each reward model"""
                # Should be the same on all ranks because the train_buf should be the same
                scales = self.label_type.target_scales(self.train_buffer.data())
                if scales is None:
                    return np.zeros([]), np.ones([])
                else:
                    return np.mean(scales, axis=0), np.std(scales, axis=0)
            self.target_mean_std = target_mean_std

            @torch.no_grad()
            def stats(query_responses):
                """return mean and std of rewards of query_responses. """
                with self.run_ctx:
                    rewards = []
                    chunk_size = query_responses[0][0].shape[0] // self.hparams.run.input_splits_reward
                    for qs, rs in query_responses:
                        rewards.append(
                            torch.cat([self.reward_model.get_rewards(q, r) for q, r in zip(qs.split(chunk_size), rs.split(chunk_size))])
                        )
                rewards = torch.cat(rewards)
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

            def set_reward_norms(mean: torch.Tensor, std: torch.Tensor, new_mean: np.ndarray, new_std: np.ndarray):
                print(f'targets: {new_mean} +- {new_std}')
                print(f'before normalize: {mean} +- {std}')
                mean, std = mean.item(), std.item()
                assert np.isfinite((mean, std, new_mean, new_std)).all()
                self.reward_model.set_reward_norm(old_mean=mean, old_std=std, new_mean=new_mean, new_std=new_std)
            self.set_reward_norms = set_reward_norms

        if self.hparams.normalize_before or self.hparams.normalize_after:

            @torch.no_grad()
            def sample_policy_batch():
                queries = query_sampler()
                chunks = queries.split(queries.shape[0] // self.hparams.run.input_splits_policy)
                with self.run_ctx:
                    responses = torch.cat([policy.respond(queries=q, length=hparams.task.response_length)['responses']
                                           for q in chunks])
                return queries, responses

            def sample_policy_responses(n_samples):
                n_batches = utils.ceil_div(n_samples, hparams.rollout_batch_size)
                return [sample_policy_batch() for _ in range(n_batches)]
            self.sample_policy_responses = sample_policy_responses

        def add_to_buffer(labels):
            return self.train_buffer.add(**labels)
        self.add_to_buffer = add_to_buffer

    def normalize(self, sample_fn, target_means, target_stds):
        """ Use target mean and std computed from human labels to set gain and bias of the reward model.
        """
        if not self.hparams.normalize_samples:
            return

        self.reset_reward_scales()
        query_responses = sample_fn(self.hparams.normalize_samples)
        means, stds = self.stats(query_responses)
        self.set_reward_norms(means, stds, target_means, target_stds)  # target mean and std are from human labels

        if self.hparams.debug_normalize:
            query_responses = sample_fn(self.hparams.debug_normalize)
            stats = self.stats(query_responses)
            self.log_stats_after_normalize(stats)

    def train(self):
        labels = download_labels(
            self.hparams.labels.source,
            schemas=self.data_schemas,
            total_labels=self.hparams.labels.num_train,
            labels_dir=self.hparams.run.labels_dir
        )
        self.add_to_buffer(labels)

        # disable gradient scaling if using bfloat16
        scaler = torch.cuda.amp.GradScaler(enabled=(self.ptdtype == torch.float16))

        if self.hparams.normalize_before:
            target_mean, target_std = self.target_mean_std()
            self.normalize(self.sample_policy_responses, target_mean, target_std)

        optimizer = self.reward_model.configure_optimizers(self.hparams)
        lr = self.hparams.lr  # for logging
        optimizer.zero_grad(set_to_none=True)  # just in case

        # we train on each point exactly once
        num_train = self.hparams.labels.num_train // params.world_size
        train_indices = params.ddp_localrank * num_train + torch.randperm(num_train)

        steps_per_batch = self.hparams.gradient_accumulation_steps * params.world_size
        assert self.hparams.batch_size % steps_per_batch == 0
        micro_batch_size = self.hparams.batch_size // steps_per_batch

        step_loss = 0  # for logging only
        total_steps = utils.ceil_div(num_train, micro_batch_size)
        for index in tqdm.trange(total_steps):
            start_index = index * micro_batch_size
            end_index = start_index + micro_batch_size
            indices = train_indices[start_index:end_index]

            minibatch = self.train_buffer.read(indices)
            for k, v in minibatch.items():
                minibatch[k] = v.to(self.hparams.run.device)

            last_micro_step = (index + 1) % self.hparams.gradient_accumulation_steps == 0
            if self.hparams.run.ddp:
                # in DDP training we only need to sync gradients at the last micro step.
                self.reward_model.set_grad_sync(last_micro_step)

            with self.run_ctx:
                stats = self.label_type.loss(reward_model=self.reward_model.get_rewards, labels=minibatch)
                loss = stats['loss'] / self.hparams.gradient_accumulation_steps

            step_loss += loss.item()
            scaler.scale(loss).backward()

            if last_micro_step:
                # clip the gradient
                # if self.hparams.grad_clip != 0.0:
                #     # need scaler.unscale_(optimizer) if using amp
                #     torch.nn.utils.clip_grad_norm_(self.reward_model.parameters(), self.hparams.grad_clip)

                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()

                # if index % self.hparams.run.log_interval == 0:
                if params.master_process:
                    step = (index + 1) // self.hparams.gradient_accumulation_steps
                    if self.hparams.run.wandb_log:
                        wandb.log({
                            "iter": step,
                            "train/loss": step_loss,
                            "lr": lr,
                        })
                    print(f"iter {step}: loss {step_loss:.4f}")
                step_loss = 0  # reset

                # schedule learning rate
                lr = (1 - index / total_steps) * self.hparams.lr
                for param_group in optimizer.param_groups:
                    param_group['lr'] = lr

        # TODO review: If we normalize after training, why are reward gain and bias trainable parameters?
        if self.hparams.normalize_after:
            target_mean, target_std = np.zeros([]), np.ones([])
            self.normalize(self.sample_policy_responses, target_mean, target_std)


def train(hparams: TrainRewardParams):

    seed = 1337 + hparams.run.seed + params.ddp_localrank
    torch.manual_seed(seed)
    np.random.seed(seed)

    if params.master_process:
        hyperparams.dump(hparams)  # output hparams to out (default to stdout)
        # download labels
        azure.download_file_cached(hparams.labels.source, hparams.run.labels_dir)

    m = trained_models.TrainedModel(None, run_hparams=hparams.run)
    encoder = m.encoding.get_encoder()

    # only used as a language model for sampling responses given context
    ref_policy = Policy(
        m, encoder,
        embed_queries=lm_tasks.query_formatter(hparams.task, encoder),
        temperature=hparams.task.policy.temperature)
    ref_policy.eval()  # no dropout

    reward_model = rewards.RewardModel(m, encoder)
    reward_model.train()

    query_batch_size = max(utils.exact_div(hparams.rollout_batch_size, params.world_size), 8)  # not too small for computing stats
    query_sampler = lm_tasks.make_query_sampler(
        hparams=hparams.task, encoder=encoder, batch_size=query_batch_size, device=hparams.run.device
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

    if params.master_process:
        reward_model.save()
        print("Trained reward model is saved.")

    if hparams.run.ddp:
        destroy_process_group()

