import torch
from torch import nn, distributions
from torch.nn import functional as F

from lm_human_preferences.language.encodings import ReversibleEncoder
from lm_human_preferences.language.trained_models import TrainedModel
from lm_human_preferences.params import TrainPolicyParams
from lm_human_preferences.utils import core_torch as utils


class Policy(nn.Module):
    def __init__(
            self,
            trained_model: TrainedModel,
            encoder: ReversibleEncoder,
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

        # model has two heads, the language model head (for policy action) and value head for state value V(s).
        self.lm_model, self.lm_params = self.trained_model.init_model('policy')  # pre-trained language model

        # Adjust this number to avoid OutOfMemoryError.
        self.micro_rollout_batch_size = 64  # make sure gradients not needed when use

    def forward(self, tokens):
        lm_output = self.lm_model(tokens, padding_token=self.encoder.padding_token)
        # need to slice logits since we don't want to generate special tokens
        logits = lm_output['lm_logits'][:, :, :self.lm_params.n_vocab]
        return {
            'logits': logits,  # shape=(b, t, n_vocab)
            'values': lm_output['hp'],  # shape=(b, t). state value V(s) where s is input to model
        }

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

        r = {'responses': [], 'logprobs': [], 'values': []}
        chunks = context.split(self.micro_rollout_batch_size) \
            if 0 < self.micro_rollout_batch_size < context.shape[0] else (context,)
        for mc in chunks:
            log_probs = []  # each item.shape=(b,). where b=mc.shape[0]
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
                logp = -F.cross_entropy(logits, next_token, reduction='none')  # shape=(b,) where b=mc.shape[0]
                log_probs.append(logp)
                # append sampled index to the running sequence and continue
                mc = torch.cat((mc, next_token[..., None]), dim=1)
            r['responses'].append(mc)  # shape=(b, context.shape[1] + length) where b=mc.shape[0]
            r['logprobs'].append(torch.stack(log_probs, dim=1))  # shape=(b, length)
            r['values'].append(torch.stack(values, dim=1))  # shape=(b, length)

        return {
            'responses': torch.cat(r['responses']),  # shape=(b, context.shape[1] + length) where b=context.shape[0]
            'logprobs': torch.cat(r['logprobs']),  # shape=(b, length)
            'values': torch.cat(r['values'])  # shape=(b, length)
        }

    def analyze_responses(self, queries: torch.Tensor, responses: torch.Tensor):
        contexts = self.embed_queries(queries)
        context_length = contexts.shape[1]
        batch, length = responses.shape
        tokens = torch.cat((contexts, responses), dim=1)  # shape=(batch, context_length + length)
        result = self(tokens)

        # context_length-1 is the first token of response
        logits = result['logits'][:, context_length-1:-1]  # shape=(batch, length, n_vocab)
        beta = 1 / torch.max(torch.as_tensor([self.temperature, 1e-10], dtype=torch.float32, device=self.device))
        logits *= beta

        # Given responses are generated by a different policy and used as labels
        logp = -F.cross_entropy(logits.view(batch * length, -1), responses.view(-1), reduction='none')  # E_p[log(q)] of logits q.
        p = F.softmax(logits, dim=-1)  # shape=logits.shape
        entropy = torch.logsumexp(logits, dim=-1) - torch.sum(p * logits, dim=-1)  # shape=(b, length)
        return dict(
            logprobs=logp.view(batch, length),  # shape=(b, length)
            entropies=entropy,  # shape=(b, length)
            values=result['values'][:, context_length-1:-1],  # shape=(b, length)
        )

    def configure_optimizers(self, hparams: TrainPolicyParams):
        device_type = 'cuda' if 'cuda' in self.device else self.device
        return self.lm_model.configure_optimizers(
            hparams.ppo.weight_decay, hparams.ppo.betas, device_type
        )
