
# lm-human-preferences

This repo is a rewrite in Pytorch 2.0 of [lm-human-preferences](https://github.com/openai/lm-human-preferences) which contains code for the paper [Fine-Tuning Language Models from Human Preferences](https://arxiv.org/abs/1909.08593), implemented using Tensorflow v1.

As I only have one single GPU with 24GB VRAM, I also made a number of changes to avoid OutOfMemoryError. Search "OutOfMemoryError" to see details. 

## Setup
Create a python environment, install packages listed in `requirements.txt`. 
My environment is Python 3.9.18, Pytorch 2.1.2.

Download pretrained checkpoint shared by OpenAI, and then run `saved_models/prepare_pytorch_checkpoint.py` to convert it to a Pytorch checkpoint.
Tensorflow 2.x is need to load the downloaded Tensorflow checkpoint, which corresponds to the language model $\rho$ to initialize reward and policy model in the paper.

OpenAI's books dataset link is broken, I use [bookcorpus](https://huggingface.co/datasets/bookcorpus) dataset hosted by hugging face.
To prepare book dataset, run `datasets/books.py`.

## Run
