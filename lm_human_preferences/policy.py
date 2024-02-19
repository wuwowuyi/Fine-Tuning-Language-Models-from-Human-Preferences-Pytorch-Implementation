import torch
from torch import nn, distributions
from torch.nn import functional as F

from lm_human_preferences.language.trained_models import TrainedModel
from lm_human_preferences.utils import core_torch as utils


class Policy(nn.Module):

    # TODO: sort out gradients and device!!!

    def __init__(
            self,
            trained_model: TrainedModel,
            encoder,
            *,
            embed_queries=lambda queries: queries,
            temperature=1.0
    ):

        super().__init__()
        self.trained_model = trained_model
        self.device = self.trained_model.device
        self.encoder = encoder
        self.embed_queries = embed_queries
        self.temperature = temperature  # used for sampling

        self.lm_model, self.lm_params = self.trained_model.init_model('policy')  # pre-trained language model

        # Adjust this number to avoid OutOfMemoryError.
        self.micro_rollout_batch_size = 8

    def forward(self, tokens):
        lm_output = self.lm_model(tokens, padding_token=self.encoder.padding_token)
        # need to slice logits since we don't want to generate special tokens
        logits = lm_output['lm_logits'][:, :, :self.lm_params.n_vocab]  # shape=(b, t, n_vocab)
        return {
            'logits': logits,
            'values': lm_output['hp'],
        }

    @torch.no_grad()
    def respond(self, queries: torch.Tensor, length: int) -> dict:
        """Given a query, sample a sequence of given `length`. """
        contexts = self.embed_queries(queries)  # shape=(b, t)
        contexts_length = contexts.shape[1]
        result = self._sample(contexts, length)
        result['responses'] = result['responses'][:, contexts_length:]
        return result

    def _sample(self, context: torch.Tensor, length: int, top_k: int = 0, top_p: float = 1.0):
        """
        Sequentially sample `length` tokens given `context`.
        :param context: context.shape=(b, t) where b is batch_size, t is length of sequence.
        :param length: number of tokens to sample sequentially
        """
        beta = 1 / torch.max(torch.as_tensor([self.temperature, 1e-10], dtype=torch.float32, device=self.device))

        assert context.shape[0] % self.micro_rollout_batch_size == 0
        r = {'responses': [], 'logprobs': [], 'values': []}
        for mc in torch.split(context, self.micro_rollout_batch_size):
            log_probs = []  # each item.shape=(b,). where b= micro_rollout_batch_size
            values = []  # each item.shape=(b,)
            for _ in range(length):
                # crop context if it's too long
                context_cond = mc if mc.size(1) <= self.lm_params.n_ctx else mc[:, -self.lm_params.n_ctx:]
                result = self(context_cond)
                logits = result['logits'][:, -1, :] * beta  # shape=(b, n_vocab)
                values.append(result['values'][:, -1])  # use last token's value
                # optionally crop the logits to only the top k options
                if top_k != 0:
                    v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                    logits[logits < v[:, [-1]]] = -float('Inf')
                if top_p != 1.0:
                    logits = utils.take_top_p_logits(logits, top_p)

                # apply softmax to convert logits to (normalized) probabilities
                dist = distributions.Categorical(logits=logits)
                # sample from the distribution
                next_token = dist.sample()  # shape=(b,)
                logp = -F.cross_entropy(logits, next_token, reduction='none')  # shape=(b,)
                log_probs.append(logp)
                # append sampled index to the running sequence and continue
                mc = torch.cat((mc, next_token[..., None]), dim=1)
            r['responses'].append(mc)  # shape=(b, context.shape[1] + length) where b=micro_rollout_batch_size
            r['logprobs'].append(torch.stack(log_probs, dim=1))  # shape=(b, length)
            r['values'].append(torch.stack(values, dim=1))  # shape=(b, length)

        return {
            'responses': torch.cat(r['responses']),
            'logprobs': torch.cat(r['logprobs']),
            'values': torch.cat(r['values'])
        }

    @torch.no_grad()
    def analyze_responses(self, queries: torch.Tensor, responses: torch.Tensor):
        contexts = self.embed_queries(queries)
        context_length = contexts.shape[1]
        batch, length = responses.shape
        tokens = torch.cat((contexts, responses), dim=1)
        result = self(tokens)
        logits = result['logits'][:, context_length-1:-1]  # shape=(b, length, n_vocab)

        beta = 1 / torch.max(torch.as_tensor([self.temperature, 1e-10], dtype=torch.float32, device=self.device))
        logits *= beta
        logp = -F.cross_entropy(logits.view(batch * length, -1), responses.view(-1), reduction='none')  # E_p[log(q)] of logits q.
        p = F.softmax(logits, dim=-1)  # shape=logits.shape
        entropy = torch.logsumexp(logits, dim=-1) - torch.sum(p * logits, dim=-1)  # shape=(b, length)
        return dict(
            logprobs=logp.view(batch, length),
            entropies=entropy,
            values=result['values'][:, context_length-1:-1],
        )
