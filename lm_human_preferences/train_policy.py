#!/usr/bin/env python3

import time
from contextlib import nullcontext
from pathlib import Path
from typing import Union

import numpy as np
import torch
import tqdm
import wandb

from lm_human_preferences import lm_tasks
from lm_human_preferences.language import trained_models
from lm_human_preferences.params import TrainPolicyParams, TaskHParams, AdaptiveKLParams
from lm_human_preferences.policy import Policy
from lm_human_preferences.rewards import RewardModel
from lm_human_preferences.utils import hyperparams, core_torch as utils


def to_numpy(tensor: Union[torch.Tensor, dict]):
    if isinstance(tensor, dict):
        return {k: to_numpy(v) for k, v in tensor.items()}
    else:
        if tensor.dtype == torch.bfloat16:  # numpy does not support bfloat16
            tensor = tensor.float()
        return tensor.to("cpu").detach().numpy()


def nupdates(hparams: TrainPolicyParams):
    return utils.ceil_div(hparams.ppo.total_episodes, hparams.ppo.batch_size)


def policy_frac(global_step: int, hparams: TrainPolicyParams):
    """How far we are through policy training."""
    return global_step / nupdates(hparams)


# def tf_times():
#     """Returns (time since start, time since last) as a tensorflow op."""
#     # Keep track of start and last times
#     with tf.init_scope():
#         init = tf.timestamp()
#
#     def make(name):
#         return tf.Variable(init, name=name, trainable=False, use_resource=True)
#
#     start = make('start_time')
#     last = make('last_time')
#
#     # Get new time and update last
#     now = tf.timestamp()
#     prev = last.read_value()
#     with tf.control_dependencies([prev]):
#         with tf.control_dependencies([last.assign(now)]):
#             return tf.cast(now - start.read_value(), tf.float32), tf.cast(now - prev, tf.float32)


class FixedKLController:
    def __init__(self, kl_coef):
        self.value = kl_coef  # beta in paper

    def update(self, current, n_steps):
        pass


class AdaptiveKLController:
    def __init__(self, init_kl_coef, hparams: AdaptiveKLParams):
        self.value: float = init_kl_coef  # beta in paper
        self.hparams = hparams

    def update(self, current, n_steps):
        target = self.hparams.target
        proportional_error = np.clip(current / target - 1, -0.2, 0.2)
        mult = 1 + proportional_error * n_steps / self.hparams.horizon
        self.value *= mult


class PPOTrainer():
    def __init__(self, *, policy: Policy, ref_policy: Policy, query_sampler, score_fn, hparams: TrainPolicyParams):
        self.policy = policy
        self.ref_policy = ref_policy
        self.score_fn = score_fn
        self.hparams = hparams

        optimizer = self.policy.configure_optimizers(hparams)

        self.ptdtype: torch.dtype = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float16
        self.run_ctx = nullcontext() if self.hparams.run.device == 'cpu' \
            else torch.amp.autocast(device_type=self.hparams.run.device, dtype=self.ptdtype)
        # disable gradient scaling if using bfloat16
        scaler = torch.cuda.amp.GradScaler(enabled=(self.ptdtype == torch.float16))

        if hparams.rewards.adaptive_kl is None:
            self.kl_ctl = FixedKLController(hparams.rewards.kl_coef)
        else:
            self.kl_ctl = AdaptiveKLController(hparams.rewards.kl_coef, hparams=hparams.rewards.adaptive_kl)

        def sample_queries():
            return query_sampler()
        self.sample_queries = sample_queries

        def compute_rewards(scores, logprobs, ref_logprobs):
            """ The per step reward, except the last, is only from KL divergence.
            The reward from reward model, `scores`, is only for the entire response, i.e., last step.
            """
            kl = logprobs - ref_logprobs  # shape=(b, length)
            non_score_reward = -self.kl_ctl.value * kl  # penalize kl divergence
            rewards = non_score_reward.detach().clone()
            rewards[:, -1] += scores  # scores.shape=(b,), add to the last step of rewards
            return rewards, non_score_reward, self.kl_ctl.value
        self.compute_rewards = compute_rewards

        minibatch_size = utils.exact_div(hparams.ppo.batch_size, hparams.ppo.nminibatches)

        def train(global_step, rollouts):
            # update learning rate for every step
            left = 1 - policy_frac(global_step, hparams)
            lrnow = hparams.ppo.lr * left
            for param_group in optimizer.param_groups:
                param_group['lr'] = lrnow

            # Do multiple epochs of PPO training, with a fresh random shuffle in each epoch
            stat_list = []
            for ppo_epoch_idx in range(hparams.ppo.noptepochs):
                order = torch.randperm(hparams.ppo.batch_size)
                for mb_start in range(0, hparams.ppo.batch_size, minibatch_size):
                    mb_data = {k: v[order[mb_start: mb_start+minibatch_size]]
                               for k, v in rollouts.items()}

                    with self.run_ctx:
                        ppo_loss, stats = self.loss(mb_data)

                    scaler.scale(ppo_loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad()

                    stat_list.append(to_numpy(stats))

            # Collect the stats. (They will be averaged later.)
            return {k: np.stack([s[k] for s in stat_list]) for k in stat_list[0].keys()}
        self.train = train

        # NOTE: must line up with stats created in self.loss (TODO: better solution?)
        def record_step_stats(*, kl_coef, **data):
            #ppo_summary_writer = utils.get_summary_writer(self.hparams.run.save_dir, subdir='ppo', comm=self.comm)

            kl = data['logprobs'] - data['ref_logprobs']
            mean_kl = np.mean(np.sum(kl, axis=1))  # sum over response_length steps, and then mean across batch
            mean_entropy = np.mean(np.sum(-data['logprobs'], axis=1))
            mean_non_score_reward = np.mean(np.sum(data['non_score_reward'], axis=1))
            stats = {
                'objective/kl': mean_kl,
                'objective/kl_coef': kl_coef,
                'objective/entropy': mean_entropy,
            }
            for k, v in data['train_stats'].items():
                stats[f'ppo/{k}'] = np.mean(v, axis=0)
            for k, v in data['score_stats'].items():
                mean = np.mean(v, axis=0)
                stats[f'objective/{k}'] = mean
                stats[f'objective/{k}_total'] = mean + mean_non_score_reward

            # stats = utils.FlatStats.from_dict(stats).map_flat(
            #     partial(utils.mpi_allreduce_mean, comm=self.comm)).as_dict()

            # Add more statistics
            stats['ppo/val/var_explained'] = 1 - stats['ppo/val/error'] / stats['ppo/returns/var']
            steps = data['global_step'] + 1
            stats.update({
                'elapsed/updates': steps,
                'elapsed/steps/serial': steps * hparams.task.response_length,
                'elapsed/steps/total': steps * hparams.ppo.batch_size * hparams.task.response_length,
                'elapsed/episodes': steps * hparams.ppo.batch_size,
            })

            # Time statistics
            #total, delta = tf_times()
            # stats.update({
            #     'elapsed/fps': tf.cast(hparams.ppo.batch_size * hparams.task.response_length / delta, tf.int32),
            #     'elapsed/time': total,
            # })
            return stats
        self.record_step_stats = record_step_stats

    def step(self, global_step: int):
        """ Like one step in environment in RL"""
        step_started_at = time.time()

        queries = self.sample_queries()  # shape=(ppo.batch_size, task.query_length). input s
        with torch.no_grad():
            with self.run_ctx:
                rollouts = self.policy.respond(queries, length=self.hparams.task.response_length)  # v(s), next_s, logP(a)
                responses = rollouts['responses']
                logprobs = rollouts['logprobs']
                rollouts['queries'] = queries

                # compute rewards
                scores, postprocessed_responses, score_stats = self.score_fn(queries, responses)
                ref_logprobs = self.ref_policy.analyze_responses(queries, responses)['logprobs']

            rewards, non_score_reward, kl_coef = self.compute_rewards(
                scores=scores,
                logprobs=logprobs,
                ref_logprobs=ref_logprobs)
            rollouts['rewards'] = rewards

        # each step t in rollout has state s_t, next state s_t+1, reward r_t, state value v(s_t), logP(a_t)
        train_stats = self.train(global_step, rollouts=rollouts)

        # now convert everything to numpy for logging
        scores, logprobs, ref_logprobs, non_score_reward, score_stats, queries, postprocessed_responses = (
            map(to_numpy, (scores, logprobs, ref_logprobs, non_score_reward, score_stats, queries, postprocessed_responses)))

        stats = self.record_step_stats(
            scores=scores, logprobs=logprobs, ref_logprobs=ref_logprobs, non_score_reward=non_score_reward,
            train_stats=train_stats, score_stats=score_stats, kl_coef=kl_coef, global_step=global_step)

        self.kl_ctl.update(stats['objective/kl'], self.hparams.ppo.batch_size)

        to_print = dict(queries=queries, responses=postprocessed_responses,
                        scores=scores, logprobs=logprobs, ref_logprobs=ref_logprobs)

        # Record profiles of the step times
        step_time = time.time() - step_started_at
        eps_per_second = float(self.hparams.ppo.batch_size) / step_time

        if global_step % 100 == 0:
            print(f"[ppo_step {global_step}] step_time={step_time:.2f}s, eps/s={eps_per_second:.2f}")
        return stats, to_print

    def loss(self, rollouts):
        values = rollouts['values']  # state values V(s)
        old_logprob = rollouts['logprobs']  # logP(action)
        rewards = rollouts['rewards']

        if self.hparams.ppo.whiten_rewards:  # whiten rewards before computing advantages
            rewards = utils.whiten(rewards, shift_mean=False)

        lastgaelam = 0
        advantages_reversed = []
        gen_length = self.hparams.task.response_length
        for t in reversed(range(gen_length)):
            nextvalues = values[:, t + 1] if t < gen_length - 1 else 0.0  # V(s_t+1)
            # TD(0) error delta = r[t] + gamma * V(s_t+1) - V(s_t)
            delta = rewards[:, t] + self.hparams.ppo.gamma * nextvalues - values[:, t]
            lastgaelam = delta + self.hparams.ppo.gamma * self.hparams.ppo.lam * lastgaelam
            advantages_reversed.append(lastgaelam)
        advantages = torch.stack(advantages_reversed[::-1], dim=1)

        # rewards are computed from per-step kl-divergence between policy and ref-policy, plus score from Reward model.
        # In RL, we usually compute returns as a discounted sum of rewards.
        # Here the author's choice is different: returns = Advantages(s,a) + V(s)
        # My speculation is it is to prevent the policy model from deviating too far away from ref-policy.
        returns = advantages + values

        # Policy is updated on every ppo_epoch and mini-batch in the loop in train()
        outputs = self.policy.analyze_responses(rollouts['queries'], rollouts['responses'])
        vpred, logprob = outputs['values'], outputs['logprobs']

        # My understanding on why vpred can be clipped this way:
        # First, V(s_t+1) is not very different from V(s_t) because inputs are one token different.
        # Second, rewards[:, :-1] = -self.kl_ctl.value * kl, reward[:, -1] = -self.kl_ctl.value * kl + (score ~ N(0, 1))
        # and then rewards are whitened.
        vpredclipped = torch.clamp(vpred, values - self.hparams.ppo.cliprange_value, values + self.hparams.ppo.cliprange_value)
        vf_losses1 = torch.square(vpred - returns)
        vf_losses2 = torch.square(vpredclipped - returns)
        vf_loss = .5 * torch.mean(torch.maximum(vf_losses1, vf_losses2))  # torch.maximum returns max value element-wise
        vf_clipfrac = torch.mean(vf_losses2 > vf_losses1, dtype=torch.float32)

        advantages = utils.whiten(advantages)  # mean center. whiten advantages before computing policy loss
        ratio = torch.exp(logprob - old_logprob)
        pg_losses = -advantages * ratio
        pg_losses2 = -advantages * torch.clamp(ratio, 1.0 - self.hparams.ppo.cliprange, 1.0 + self.hparams.ppo.cliprange)
        pg_loss = torch.mean(torch.maximum(pg_losses, pg_losses2))
        pg_clipfrac = torch.mean(pg_losses2 > pg_losses, dtype=torch.float32)

        # NOTE: the entropy bonus is not added to the objective, like the PPO paper does.
        # I understand it is because KL-divergence is included in rewards.
        loss = pg_loss + self.hparams.ppo.vf_coef * vf_loss

        # for debugging/logging
        entropy = torch.mean(outputs['entropies'])
        approxkl = .5 * torch.mean(torch.square(logprob - old_logprob))
        return_var, return_mean = torch.var_mean(returns, dim=list(range(returns.dim())), correction=0)
        value_var, value_mean = torch.var_mean(values, dim=list(range(values.dim())), correction=0)
        stats = dict(
            loss=dict(policy=pg_loss, value=vf_loss, total=loss),
            policy=dict(entropy=entropy, approxkl=approxkl, clipfrac=pg_clipfrac),
            returns=dict(mean=return_mean, var=return_var),
            val=dict(vpred=torch.mean(vpred), error=torch.mean((vpred - returns) ** 2),
                     clipfrac=vf_clipfrac, mean=value_mean, var=value_var)
        )
        return loss, utils.flatten_dict(stats, sep='/')


def make_score_fn(hparams: TaskHParams, score_model: RewardModel):
    padding_token = score_model.padding_token

    # tokens after truncation are set to padding token
    postprocess = lm_tasks.postprocess_fn_from_hparams(hparams, padding_token)
    # ensure that the sample contains truncate_token
    filter_fn = lm_tasks.filter_fn_from_hparams(hparams)

    def penalize(responses, rewards):
        valid = filter_fn(responses)
        return torch.where(valid, rewards, hparams.penalty_reward_value * torch.ones_like(rewards))

    @torch.no_grad()
    def unpenalized_score_fn(queries, responses):
        # unpenalized score ~ N(0, 1)
        return score_model.get_rewards(queries, responses)

    def score_fn(queries, responses):
        responses = postprocess(responses)
        score = penalize(responses, unpenalized_score_fn(queries, responses))
        return score, responses, dict(score=score)
    return score_fn


def log_samples(encoder, hparams: TrainPolicyParams, to_print: dict):
    queries, responses, scores = to_print['queries'], to_print['responses'], to_print['scores']
    logprobs, ref_logprobs = to_print['logprobs'], to_print['ref_logprobs']

    # Log samples
    for i in range(min(3, len(queries))):
        sample_kl = np.sum(logprobs[i] - ref_logprobs[i])
        wandb.log({
            "queries": str(encoder.decode(queries[i][:hparams.task.query_length]).replace("\n", "⏎")),
            "responses": str(encoder.decode(responses[i]).replace("\n", "⏎")),
            "score": scores[i],
            "kl": sample_kl,
            "total": scores[i] - hparams.rewards.kl_coef * sample_kl
        })


def train(hparams: TrainPolicyParams):

    seed = 1337 + hparams.run.seed
    torch.manual_seed(seed)
    np.random.seed(seed)

    save_dir: Path = hparams.run.save_dir
    assert save_dir is not None, "save_dir cannot be None!"
    assert Path(hparams.rewards.trained_model).is_file(), "Reward checkpoint does not exist. Please train reward first."

    hyperparams.dump(hparams)

    m = trained_models.TrainedModel(hparams.task.policy.initial_model, run_hparams=hparams.run)
    encoder = m.encoding.get_encoder()

    # if save_dir:
    #     if not save_dir.startswith('https:'):
    #         os.makedirs(os.path.join(save_dir, 'policy'), exist_ok=True)
    #     with tf.io.gfile.GFile(os.path.join(save_dir, 'train_policy_hparams.json'), 'w') as f:
    #         json.dump(hparams.to_nested_dict(), f, indent=2)
    #     with tf.io.gfile.GFile(os.path.join(save_dir, 'policy', 'hparams.json'), 'w') as f:
    #         json.dump(m.hparams().to_nested_dict(), f, indent=2)
    #     with tf.io.gfile.GFile(os.path.join(save_dir, 'policy', 'encoding'), 'w') as f:
    #         json.dump(m.encoding.name, f, indent=2)

    ref_policy = Policy(
        m, encoder,
        embed_queries=lm_tasks.query_formatter(hparams.task, encoder),
        temperature=hparams.task.policy.temperature)
    ref_policy.train()  # we don't train ref_policy, setting to train mode so it behaves identically to policy

    policy = Policy(
        m, encoder,
        embed_queries=lm_tasks.query_formatter(hparams.task, encoder),
        temperature=hparams.task.policy.temperature)
    policy.train()

    query_sampler = lm_tasks.make_query_sampler(
        hparams=hparams.task, encoder=encoder, batch_size=hparams.ppo.batch_size, device=hparams.run.device
    )

    minibatch_size = utils.exact_div(hparams.ppo.batch_size, hparams.ppo.nminibatches)
    if hparams.ppo.whiten_rewards:
        assert minibatch_size >= 8, \
            f"Per-rank minibatch size {minibatch_size} is insufficient for whitening"

    m.initial_model = hparams.rewards.trained_model
    score_model = RewardModel(m, encoder)
    score_model.eval()

    ppo_trainer = PPOTrainer(
        policy=policy, ref_policy=ref_policy, query_sampler=query_sampler,
        score_fn=make_score_fn(hparams.task, score_model=score_model),
        hparams=hparams)

    policy.save()

    try:
       for global_step in tqdm.trange(nupdates(hparams)):
            stats, to_print = ppo_trainer.step(global_step)

            if hparams.run.wandb_log and global_step % hparams.run.log_interval == 0:
                wandb.log(stats)  #TODO: review
                log_samples(encoder, hparams, to_print)

            if global_step % hparams.run.save_interval == 0:
                policy.save()

    finally:
        policy.save()
