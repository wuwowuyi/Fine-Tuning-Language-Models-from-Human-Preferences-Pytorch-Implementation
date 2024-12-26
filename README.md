
# lm-human-preferences

This repo is my rewrite in Pytorch 2.0 of [lm-human-preferences](https://github.com/openai/lm-human-preferences) which is Tensorflow v1 implementation of the paper [Fine-Tuning Language Models from Human Preferences](https://arxiv.org/abs/1909.08593).

This paper is OpenAI’s original RLHF work from 2019. See [my notes of this paper](Paper.md). 

In addition to a rewrite in Pytorch 2.0, other changes I made:
* Removed all the MPI code, and use [Pytorch DDP](https://pytorch.org/tutorials/intermediate/ddp_tutorial.html) to train on multiple GPUs.
* Made a number of changes to avoid OutOfMemoryError. Search "OutOfMemoryError" to see details. The default settings fit a 24GB GPU.
* Use bfloat16 and [torch.amp package](https://pytorch.org/docs/stable/amp.html) to speed up training

## Methods

### Reward
The reward model is formulated as a classification model in the same way as a language model.

A linear layer on top of the underlying transformer converts output embeddings to a scalar reward which is treated as raw logit of a Categorical distribution, with the best response as the "true" label. 

Specifically, for a data point tuple $(x, y_0, y_1, y_2, y_3, b)$, $x$ is concatenated respectively with $y_0$, $y_1$, $y_2$ and $y_3$ as input, and the reward model computes 4 scalar rewards $r_0, r_1, r_2, r_3$ which are treated as raw logits of a 4-class Categorical distribution with the best response $b \in \{0, 1, 2, 3\}$ as the true label. And then we compute the cross entropy loss.

Here $x$ corresponds to the state $s$ in Reinforcement Learning (RL), and $y_0$, $y_1$, $y_2$ and $y_3$ are seen as different actions $a$. Therefore, the reward model gives a scalar reward $r$ for each $(s, a)$ pair.

### Policy

The underlying transformer is shared by the language model and state value function.

The language model itself is a policy actor. At each step, the policy generates the next token, i.e., action.

To compute advantage, we need a state value function $V(s)$. This is implemented by using a linear layer on top of transformer to convert output embeddings to state values, where state $s$ is the input.

## Training
There is no online data collection, so training is offline,

Notations:
* $\rho$ pretrained language model
* $R$ reward model, initialized from $\rho$
* $\pi$ policy model, initialized from $\rho$
* $\pi_{ref}$ reference policy, initialized from $\rho$. no training on it, so $\pi_{ref} = \rho$.
* human labelled dataset $S$, like descriptiveness offline 5k, each data point is a $(x, y_0, y_1, y_2, y_3, b)$ tuple
* dataset $D$, such as the bookcorpus dataset
* $x$ query, ie., prompt
* $y$ response

### Reward Training
Reward training is formulated as a classification problem where the labels are the best responses picked by human labelers.

* First, initialize $R$ and $\pi$. Here $\pi$ is only used a language model to generate responses to queries.
* Second, download $S$ and store in `SampleBuffer`.
* Third, set rewards gain $g_r$ and bias $b_r$ of $R$ so that the output rewards will have roughly mean 0 and variance 1
  * sample a batch of $x$ from $D$
  * ask $\pi$ to generate responses $y$, ie, $y \sim \pi(y|x)$
  * ask $R$ to compute rewards for each pair of $(x, y)$ to get rewards mean and variance
  * set $g_r$ and $b_r$ to have zero mean and standard deviation
* Fourth, train one epoch of $R$ on $S$.
* Lastly, repeat the third step to update $g_r$ and $b_r$ so that output reward $\sim N(0, I)$.

### Policy Training

* Load trained $R$, initialize $\pi$ and $\pi_{ref}$.
* sample a mini-batch of rollouts
  * $\pi$ generates given length of $y$ for batched $x$ sampled from $D$, ie, $y \sim \pi(y|x)$, $x \sim D$
  * each $y$ is post processed by replacing all tokens after the first `truncate_token` after number of `truncate_after` tokens with the the padding token. The `truncate_after` is, for example, 16 for sentiment task, and 55 for cnndm. `truncate_token` is the period `.` or `\n`.
  * if $y$ contains no `truncate_token`, the reward is `-1`, otherwise $R$ computes a reward for $(x, y)$.
  * computes $\log \pi_{ref}(y|x)$ by feeding $(x, y)$ into the policy reference model
  * compute per step reward $r_t = -\beta \cdot D_{KL}(\pi||\pi_{ref}) = -\beta \cdot (\log \pi(y|x) - \log \pi_{ref}(y|x))$
  * add rewards computed from $R$ onto the last step
  * as a result, the rollouts contains a collection of queries $x$, responses $y$, rewards $r$, state values $V(x, y)$, and $\log\pi(y|x)$, each with `shape=(batch_size, query_length)`. The batch size can be as low as 8 on one GPU.
* compute `loss = pg_loss + vf_coef * vf_loss`, where `pg_loss` is the policy loss, `vf_loss` is the state value loss, and hyperparameter `vf_coef` is a coefficient with default value 0.1
  * first compute advantages, as described in the [PPO paper](https://arxiv.org/abs/1707.06347). (details can also see [GAE paper](https://arxiv.org/abs/1506.02438))
  * `pg_loss` is computed as described in the PPO paper.
  * `loss` contains a value function error term `vf_loss =` $(V_\theta(s_t) - V_t^{targ})^2$ because policy and state value share the underlying transformer
    * where $V_t^{targ}$ `= advantages + values` where `values` are from the sampled rollouts. Entropy bonus described in the paper is contained in `advantages`.
    * clipping is also used to prevent `vf_loss` from becoming too small, because smaller loss would allow model parameters to drift farther.
  

## Experiment Setup
Create a python environment, install packages listed in `requirements.txt`. 
My environment is Python 3.9, Pytorch 2.1.2.

### Preparation
#### Prepare Pytorch checkpoint for pretrained language model
Download a pretrained tensorflow checkpoint shared by OpenAI, and then run `saved_models/prepare_pytorch_checkpoint.py` to convert it to a Pytorch checkpoint.
Tensorflow is needed to load the downloaded checkpoint, which corresponds to the language model $\rho$ to initialize reward and policy model in the paper.

Details see `saved_models/prepare_pytorch_checkpoint.py`.

#### Prepare datasets
OpenAI's books dataset link is broken, I use [bookcorpus](https://huggingface.co/datasets/bookcorpus) dataset hosted by Hugging face.
To prepare book dataset, run `datasets/books.py`.

To prepare cnndm dataset ([cnn daily mail](https://huggingface.co/datasets/abisee/cnn_dailymail), also from Hugging face), run `datasets/cnndm.py`.

To prepare for tldr dataset, run `datasets/tldr.py` which will download and process OpenAI's tldr dataset hosted on Azure.

The generated datasets are located under `datasets`, for example, `bookcorpus_train.bin`.

### Run

This project uses [wandb](https://wandb.ai/) for logging, run `wandb login` to login first.

First step is to train a reward model. 
`launch.py` is the entry point for training. It requires at least two arguments to train reward:
* task name. Four tasks are supported, ie, sentiment, descriptiveness, cnndm, and tldr.
* experiment name, ie., the name for a particular run, can be anything.

For example:
```commandline
experiment=descriptiveness
reward_exp=desc_reward
python ./launch.py train_reward $experiment $reward_exp
```

After training reward, we can then train a policy model. 
In addition to task name and experiment name, must provide path to the checkpoint of trained reward model.
For example:
```commandline
experiment=descriptiveness
policy_exp=desc_policy
trained_reward=$(pwd)/lm_human_preferences/saved_models/descriptiveness_reward_ckpt.pt
python ./launch.py train_policy $experiment $policy_exp --rewards.trained_model $trained_reward --rewards.train_new_model 'off'
```

Task specific parameters are in `launch.py`. I also collected all the experiment parameters in `params.py`.
To override default training parameters (task and experiment), read function `get_params` in `utils/launch.py`.

## Results

### tldr
![tldr policy training return](/static/tldr_ppo_policy.png)

The last generated query and response
```text
5153 queries: This is quite a convoluted issue, but I'm hoping someone will be able to help because Riot support has stopped responding to me.⏎>⏎>Basically, in December I was charged twice for some RP I bought, as I was told that the first payment was declined by the store so I paid by a different method, only to find on my statement that in fact both payments had gone through. I only received one payment's worth of RP. I contacted Riot support about this in early January, as was advised to take out a claim against the transaction with my bank, using a transaction ID provided to me by SplendidClaw.⏎>Recently, I received notice from my bank that the refund had been successful, and I got the money back. However, I also received an email from Riot on the same day telling me that my account had been disabled and I needed to  give them back the money  in order to get my account back.⏎>Now all I receive from them are automated messages on him to repay them the money that they shouldn't ever have received. It seems to be a ridiculous situation where both parties think they are owed money by the other, especially seeing as all I did was follow the advise given to me.⏎>⏎>Has anyone else ever been in this situation? What can I do?⏎>I've invested a lot of time and money into that account so I don't really want to see it binned!⏎>⏎>

5154 responses:  In December I was charged twice for some RP I bought, as I was told that the first payment was declined by the store so I paid by a different method, only to find on my statement that in fact both payments had gone through. I only received one payment's worth of RP. I contacted Riot support about this in early January, as was advised to take out⏎
```

### descriptiveness

![descriptiveness policy training return](/static/descriptive_ppo_policy.png)

The last 3 generated queries and responses.
```text
1427 queries: fiddling with the conundrum that was the dress i had on to pass time , i gave up .
1428 responses:  The dress was perfect for me and my leg was perfectly straight and very light. I was able to wear it straight.
1429 queries: `` just as long as you do n't read the garbage they print with the pictures , we 'll be fine . ''
1430 responses: ⏎⏎The void thing is that it is very easy to tell if the object is garbage or not.
1431 queries: his eyes sparkled as he squeezed her hand .
1432 responses:  He looked up, a set of white eyes, and a tinge of pride, on his brow.
```

### Sentiment
```text
583 queries: i replied enthusiastically .
584 responses:  "Good evening to you, Eliza," said Nail, with a smile.
585 queries: `` i want open fighting right here in the halls of the palace .
586 responses:  .! ! ! ~~~~ ~~~⏎⏎I want to start a discussion on this topic.
587 queries: beyond the next turn awaited a dead end .
588 responses: ⏎⏎⏎It was a huge project that was done in many different ways. Each man is different.
```

### cnndm
![cnndm policy training return](/static/cnndm_ppo_policy.png)
