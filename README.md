# Tweet-Sentiment-Extraction
This repository is my semester project in course Intelligent Data Analysis &amp; Machine Learning II in winter semester 2022/23 at the University of Potsdam.
The project provides a solution to the Kaggle [Tweet Sentiment Extraction](https://www.kaggle.com/competitions/tweet-sentiment-extraction/overview) Competition. 

## Problem setting & dataset
### Problem setting
For a given tweet, predict what word or phrase best supports the sentiment labels (positive, negative, neutral). The word or phrase should include all characters within that span (i.e. including commas, spaces, etc.).

### Dataset
The dataset of the competition can be downloaded from the [Kaggle website](https://www.kaggle.com/competitions/tweet-sentiment-extraction/data). 

Needed files: \
train.csv - the training set \
test.csv - the test set \
sample_submission.csv - a sample submission file in the correct format 

Columns: \
```textID``` - unique ID for each piece of text \
```text``` - the text of the tweet \
```sentiment``` - the general sentiment of the tweet \
```selected_text``` - [train only] the text that supports the tweet's sentiment 

Disclaimer: The dataset for this competition contains text that may be considered profane, vulgar, or offensive.


## Approaches
### Modelling as a Question & Answering (Q&A) problem
Given a question and a context, train a transformer model that is able to find the answer in the context.

We have: \
Question: ```sentiment``` column (positive/negative/neutral) \
Context: ```text``` column \
Answer: ```selected_text``` column 

### Models
[deepset/roberta-base-squad2](https://huggingface.co/deepset/roberta-base-squad2) (roberta-base model, fine-tuned using the SQuAD2.0 dataset) \
[distilbert-base-cased-distilled-squad](https://huggingface.co/distilbert-base-cased-distilled-squad) (DistilBERT-base-cased, fine-tuned using (a second step of) knowledge distillation on SQuAD v1.1) 

## Evaluation metric
The metric in this competition is the [word-level Jaccard score](https://en.wikipedia.org/wiki/Jaccard_index). The Jaccard score is defined as the size of the intersection divided by the size of the union of the sample sets:
``` math
Jaccard(U,V) = \frac{|U \cap V|}{|U \cup V|}
```

## Results
Fine-tuned roberta-base-squad2: **70,2** Jaccard score \
Fine-tuned distilbert-base-cased-distilled-squad: **68,6** Jaccard score 
