## SQuAD DeBERTa performance analysis

## Project Overview

I started to be curious about Question-Answering tasks and the first dataset I started with was [SQuAD v2.0](https://rajpurkar.github.io/SQuAD-explorer/) and the model I used in this project was [DeBERTa v3 small](https://huggingface.co/microsoft/deberta-v3-small) by Microsoft.

I also read some papers about how Machine Learning models often tend to learn shortcuts [[References](2a48774b9a3e3bb7af80764a8f326bc4)]. My mentor suggested that increasing the negative samples in the training dataset could potentially improve the model's performance, so I experimented with different ways to sample more negative samples.

## Negative sample sampling methods

The most important thing is understanding what *negative samples* are.

Basically, each SQuAD sample has a structure like this:



### 1. Weighted Sampling 



### 2. Question classification 



### 3. Replacing keywords



### References: 

1. [Why Machine Reading Comprehension Models Learn Shortcuts?](https://arxiv.org/pdf/2106.01024.pdf)
2. [Do We Know What We Don’t Know? Studying Unanswerable Questions beyond SQuAD 2.0](https://aclanthology.org/2021.findings-emnlp.385.pdf)
