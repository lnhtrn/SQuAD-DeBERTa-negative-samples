## SQuAD DeBERTa Performance with Increased Negative Samples

## Project Overview

The project is about improving [DeBERTa v3 small](https://huggingface.co/microsoft/deberta-v3-small) MRC/Question-Answering model by Microsoft using [SQuAD v2.0](https://rajpurkar.github.io/SQuAD-explorer/) dataset.

The source code for model training can be found in the [/src](src) folder with explanations, and can be run using the scripts in the [/scripts](scripts) folder. The results are stored in the [/results](results) folder with a notebook detailing the analysis process.

Additional requirements outside of [requirements.txt](requirements.txt) might be needed to run the project. 

---

## Introduction

After knowing that Machine Learning models often tend to learn shortcuts[^1][^2], I tried to brainstorm different ways of reducing shortcuts to apply in my own project. 

![An illustration of shortcuts in Machine Reading Comprehension. P is an excerpt of the original passage.](https://github.com/lnhtrn/SQuAD_DeBERTa_performance_analysis/assets/72944083/ac68e209-a25a-4c6a-8d77-de6ae90d4aab)   
<sub>*An illustration of shortcuts in Machine Reading Comprehension*</sub>[^1]

My mentor suggested that increasing the negative samples in the training dataset could potentially improve the model's performance, so I experimented with different ways to sample more negative samples.

[^1]: [Why Machine Reading Comprehension Models Learn Shortcuts?](https://arxiv.org/pdf/2106.01024.pdf)

[^2]: [Do We Know What We Don’t Know? Studying Unanswerable Questions beyond SQuAD 2.0](https://aclanthology.org/2021.findings-emnlp.385.pdf)



## Negative sample sampling methods

Each SQuAD sample has a structure like this:

```json
{
   "question": "When did Beyonce start becoming popular?",
   "id": "56be85543aeaaa14008c9063",
   "answers": [
      {
         "text": "in the late 1990s",
         "answer_start": 269
      }
   ],
   "is_impossible": false
}
```

A *positive sample* is a sample with `context`, `question`, and one or more correct `answers` - which means `is impossible = false`. A `negative sample`, respectively, is a sample where `is impossible = true`, the `answers` list is empty, and there might be a `plausible_answers` folder. An example of a negative sample is:

```json
{
   "plausible_answers": [
      {
         "text": "action-adventure",
         "answer_start": 128
      }
   ],
   "question": "What category of game is Legend of Zelda: Australia Twilight?",
   "id": "5a8d7bf7df8bba001a0f9ab1",
   "answers": [],
   "is_impossible": true
}
```

This project aims to generate more `negative sample`s like this to see if that would improve the model's performance.



### 1. Weighted Sampling 

The original distribution of positive and negative samples in the training set was roughly 50:50. I used Weighted Sampling (`torch`'s `WeightedRandomSampler`) to generate different ratios of positive and negative samples in the training set to see if there would be any improvement in model accuracy.




### 2. Generated Samples: Question classification 

All questions in the training dataset are classified into 7 types (What, Who, Where, Why, When, Which, Count) before this process using a classification model. 

The data was first scanned and classified into 7 categories based on the first question word, and those whose first word does not explicitly belong to the 7 categories were labeled `Unlabeled`. The classification model was trained on the labeled data and was used to predict the unlabeled data.

Then, I generated new questions by choosing questions with a similar type to the original questions, but replacing some nouns with other nouns matching the context.

The newly generated question has no correct answer, thus making it a negative sample. 

![Example of using classification to generate new questions](https://github.com/lnhtrn/SQuAD_DeBERTa_performance_analysis/assets/72944083/eb0e39ce-33c5-475c-bf92-0149c0c96149)






### 3. Generated Samples: Replacing keywords

First, I used `transformer`'s NER pipeline (`pipeline("ner", aggregation_strategy='average')`) on contexts. Then, I generated new questions based on existing questions for the same context, replacing certain words in the question with other keywords found in the text. 

The newly generated question has no correct answer, thus making it a negative sample. 

![Example of replacing keywords to create new questions](https://github.com/lnhtrn/SQuAD_DeBERTa_performance_analysis/assets/72944083/4180f345-ef5f-4150-8e7d-6e24afe51de7)


## Model training

This is explained in the [/src](src) folder's `readme` file. 



## Results

This is the overall results between all models:

![image](https://github.com/lnhtrn/SQuAD-DeBERTa-negative-samples/assets/72944083/aa91cdee-e811-4793-a112-1453f07528f5)

### 1. Weighted Sampling method

### 2. Swap method

### 3. Replace method

The model trained with augmented data based on replace method does better when evaluating on the original dataset than on the augmented dev dataset. 

![image](https://github.com/lnhtrn/SQuAD_DeBERTa_performance_analysis/assets/72944083/9f0fdf28-7398-4680-a3a8-f04366856d12)

Changing the ratio of positive/negative samples affects the model efficiency as well. 

Increasing the percentage of negative samples seems to make the model behave more inconsistently - the difference between the accuracy of answerable questions and unanswerable questions increases while the overall accuracy of the model decreases. 

However, setting the percentage of the positive samples too high caused the model to decrease its efficiency again.

![image](https://github.com/lnhtrn/SQuAD_DeBERTa_performance_analysis/assets/72944083/e85e83c1-c104-422c-8b8f-3786b8312d5e)



### References: 

[^1]: [Why Machine Reading Comprehension Models Learn Shortcuts?](https://arxiv.org/pdf/2106.01024.pdf)

[^2]: [Do We Know What We Don’t Know? Studying Unanswerable Questions beyond SQuAD 2.0](https://aclanthology.org/2021.findings-emnlp.385.pdf)
