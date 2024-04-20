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

In the online data collection case, we continue to collect additional samples, and periodically retrain the reward $r$, and then optimize $\pi$ against $r$.

### Pretraining details
We use a 774M parameter version of GPT-2 trained on WebText dataset and 50,257 token invertible BPE tokenizer.

For stylistic continuation tasks (sentiment and descriptiveness), **we perform supervised finetuning on BookCorpus dataset prior to RL fine-tuning**.

To improve sample quality, we use a temperature T < 1 for all experiments.

### RL fine-tuning details
**The reward model** is trained using Adam, batch size 8 for style tasks and 32 for summarization. 
The learning rate is $1.77 \times 10^{-5}$ for both.
We use a single epoch to avoid overfitting to the small amount of human data, and turn off dropout. 

**On training policy**, we use 2M episodes, $\gamma = 1$, 4 PPO epochs per batch.
We use batch size 1024 for style tasks and 512 for summarization.
We do not dropout for policy training.
The learning rate was $1.41 \times 10^{-5}$ for style tasks, and $7.07 \times 10^{-6}$ for summarization.

Models trained with different seeds and the same KL penalty $\beta$ sometimes end up with quite different values of $KL(\pi, \rho)$, making them hard to compare. 
To fix this, **for some experiments we dynamically vary $\beta$ to target a particular value of $KL(\pi, \rho)$** using log-space proportional controller:
$e_t = clip(\large{\frac{KL(\pi_t, \rho) - KL_{target}}{KL_{target}}}, -0.2, 0.2)$, 
$\beta_{t+1} = \beta_t(1+K_{\beta}e_t)$, and we used $K_{\beta} = 0.1$.
i.e. when at an iteration step $t$, the $KL(\pi_t, \rho)$ is larger than a target KL penalty, $\beta_{t+1}$ for the next iteration is increased, otherwise decreased.

**For supervised fine-tuning baseline**, we finetune for 1 epoch on the CNN/Daily Mail and TL;DR training sets. For TL;DR we removed 30K examples to serve as a validation set. Training use cosine schedule, initial learning rate was selected by sweep over 8 log-linearly spaced options between $10^{-4}$ and $3 \times 10^{-4}$.
We found dropout 0.1 to work best. We then choose the model with the best validation loss.

### Online Data collection
**If the trained policy $\pi$ is very different from the zero-shot policy $\rho$, the reward model will suffer a large distributional shift from training on samples from $\rho$ to evaluation on samples from $\pi$**. To prevent this, we can collect human data throughout RL fine-tuning, continuously gathering new data by sampling from $\pi$ and retraining the reward model.
As experiments shows, **online data collection was important for summarization but not for the simpler style tasks**.

For a total of 2M PPO episodes, we train the reward model before the first PPO episode, and then retrain it 19 more times at evenly spaced values.

Each time we retrain we reinitialize $r$  to a random linear layer on top of $\rho$ and do a single epoch through the labels collected so far. (NOTE: The authors choose to init from $\rho$, even when retrain it, and with all labels. Maybe if the reward model was initiated from $\pi$, the bias in $\pi$ would be reinforced. Further, in this way the validation samples can be used too, see below.)

To estimate overall progress, we gather validation samples consisting of $x \sim D; y_0, y_1 \sim \rho(\cdot|x); y_2, y_3 \sim \pi(\cdot|x)$ at a constant rate. Human labels on these give how often $\pi$ beats $\rho$. Since validate samples are only used to evaluate the current $\pi$, we can add them to the training set for $r$.

### Human labeling

There is significant disagreement among labelers, and between labelers and authors. Specifically, the authors agree with Scale AI labelers 38% of the time on sentiment and 46% of the time on TL;DR summarization.

Evaluations show $r$ and $\pi$ can successfully fit to human reward, but does not show that those human evaluations capture what we really care about, and our models are incentivized to exploit idiosyncracies of the labeling process.

## Experiments
### Stylistic continuation tasks: sentiment and descriptiveness
$x$ is sampled from BookCorpus dataset with a length of 32 to 64 tokens, and policy generates 24 tokens in response. Temperature is 0.7.

#### Mock sentiment task
Train a reward function $r_s$ by training a classifier (a transformer with 6 layers, 8 attention heads, embed size 512) on a binarized, balanced sumsample of Amazon Review dataset. $r_s(x, y)$ is the logP of a review being positive. (i.e., encourage the policy to generate positive reviews)

![mock sentiment learning curve (KL 8)](/static/mock-curves.svg)
where direct means direct RL access to $r_s$.

Because we know the reward function $r_s$, we can also analytically compute the optimal policy. The optimal policy has the form: $\pi_{opt}(y|x) \propto \rho(y|x)e^{r_s/\beta}$

We approximate the reward of this policy for given $x$ and $\beta$ by sampling a large number of y from $\rho$ and re-weighting them by $e^{r_s/\beta}$. see figure below.

![KL vs mock reward](/static/mock-kl-frontier.svg)

#### Human evaluations
Sentiment task: encourage "positive and happy" continuations
Descriptiveness task: encourage "vividly descriptive" continuations

x is sampled from Bookcorpus, starting and ending with a period. Rejection sampling is used to ensure there is a period between continuation tokens 16 and 24 and then truncate at that period. During RL fine-tuning, we penalize continuations that don't have such a period by giving a fixed reward of -1.

We dynamically adjust $\beta$ to obtain a KL of 6 nats for descriptiveness and 10 nats for sentiment.
![sentiment compare to ref](/static/sentiment_compare_to_ref.svg)

![descriptiveness compare to ref](/static/descriptiveness_compare_to_ref.svg)

![human evaluations for continuations](/static/human%20evaluations%20for%20continuations.png)















