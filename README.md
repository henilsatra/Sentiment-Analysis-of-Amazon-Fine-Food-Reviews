# Project Title

Final Project for CIS 5190

## Description

For this project, we study the problem of sentiment analysis of product reviews, which is important for ecommerce sites in achieving higher customer satisfaction and sales. A positive review has a rating of 4 or 5 and a negative review has a rating of 1, 2 or 3. Text cleaning and feature extraction techniques were performed on the reviews data, namely removing special characters and numbers, removing stop words, and tokenizing the data. Other forms of text vectorizations were also examined, including Word2Vec, GLOVE and Bag-of-Words. We developed baseline models using simple machine learning classifiers, like Naive Bayes, Logistic Regression, and XGBoost, before moving to Long short-term memory (LSTM) models. Finally we also evaluated BERT embeddings with LSTM. We evaluated our models in binary classification tasks through different performance metrics, namely through computing the confusion matrix, the AUC score, and the F1 score. From our baseline results, we observed that Bag-of-Words feature extraction obtained the best performance on the original dataset and the dataset shift. We also observe that LSTM+BERT performs the best on the training data followed by LSTM+GLOVE and then LSTM+Word2Vec. The number of dimensions of features play a big role in extracting information. We see they do not perform very well in baselines as taking an average loss of the information extracted. 

## Getting Started

### Dependencies

* Google Colaboratory with Nvidia CUDA GPU 

### Installing

#### Importing Amazon Fine Food Reviews Dataset 

```
! pip install --upgrade --no-cache-dir gdown
if not os.path.exists("Reviews.csv"):
    !gdown 1_kLSwiRYtiXF7h9V1FlTqTapOHiYU5Mk
```

#### Importing Large Movie Review Dataset 
```
wget http://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz
tar -xf aclImdb_v1.tar.gz
```

### Executing program

* How to run the program
* Step-by-step bullets
```
code blocks for commands
```

## Help

Any advise for common problems or issues.
```
command to run if program contains helper info
```

## Authors

Contributors names and contact info

ex. Dominique Pizzie  
ex. [@DomPizzie](https://twitter.com/dompizzie)
