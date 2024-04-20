# Paper Summarization

## Introduction
Humans need to specify goals to AI agents using natural language, and AI agents need to communicate back to a human using natural language too.

There is a long literature applying Reinforcement Learning (RL) to natural language tasks. Much of this work uses algorithmically defined reward functions, and some use human evaluations.

We refer to [Luketina et al. (2019)](https://arxiv.org/abs/1906.03926) for a survey of RL tasks involving language as a component, and for RL results using transfer learning from language.

【总结完全文再来看 introduction 部分的实验总结】

## Methods
We will first use human labels to train a reward model, and then optimize this reward model.

### Train reward and policy
Following [Christiano et al. (2017)](https://arxiv.org/abs/1706.03741), we ask human labelers to pick which of several values of $y_i$ is the best response to a given input $x$. (In early experiments, we found it was hard for humans to provide consistent fine-grained quantitative distinctions when asked for an absolute number, and experiments on synthetic tasks confirmed that comparisons were almost as useful.)

Let $b \in \{0, 1, 2, 3\}$ be the option human labelers select from four options $(y_0, y_1, y_2, y_3)$. Having collected a dataset $S$ of $(x, y_0, y_1, y_2, y_3, b)$ tuples, we fit a reward model $r: X \times Y \to R$ using the loss:
$loss(r) = -E_{(x, \{y_i\}_i, b)\sim S}[log\Large{\frac{e^{r(x, y_b)}}{\sum_ie^{r(x, y_i)}}}]$
(Note: the original paper does not have the minus sign. But as a loss, there should be a minus sign here, i.e., negative log probability)

The reward model is initialized as a random linear function on top of a language model $\rho$.
To keep the scale of the reward model consistent across training, we normalize it so that it has mean 0 and variance 1 for $x \sim D, y \sim \rho(\cdot|x)$.

We also initialize a policy $\pi = \rho$, and finetune it to optimize the reward model $r$ via Proximal Policy Optimization (PPO) with a modified reward: $R(x, y)=r(x, y) - \beta log\large{\frac{\pi(y|x)}{\rho(y|x)}}$

The term $\beta log\large{\frac{\pi(y|x)}{\rho(y|x)}}$ acts as an entropy bonus (i.e. $-log\pi$) and KL penalty (i.e. $log\pi - log\rho$) to prevent $\pi$ from moving too far away from $\rho$.

In the online data collection case, we continue to collect additional samples, and periodically retrain the reward r, and then optimize $\pi$ against r.

### Pretraining details
We use a 774M parameter version of GPT-2 trained on WebText dataset and 50,257 token invertible BPE tokenizer.

For stylistic continuation tasks (sentiment and descriptiveness), **we perform supervised finetuning on BookCorpus dataset prior to RL fine-tuning**.

To improve sample quality, we use a temperature T < 1 for all experiments.

### RL fine-tuning details
The reward model is trained using Adam, batch size 8 for style tasks and 32 for summarization. 
The learning rate is $1.77 \times 10^{-5}$ for both.
**We use a single epoch to avoid overfitting to the small amount of human data, and turn off dropout**. 

On training policy, we use 2M episodes, $\gamma = 1$, 4 PPO epochs per batch.
We use batch size 1024 for style tasks and 512 for summarization.
We do not dropout for policy training.
The learning rate was $1.41 \times 10^{-5}$ for style tasks, and $7.07 \times 10^{-6}$ for summarization.






