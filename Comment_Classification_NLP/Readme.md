## Delay root cause inference based on the comment documents, using natural language processing. 

In this project, the comments from schedulers were classified into 11 categories using natural language processing.

## Description of the dataset:
The dataset consists of a table with 1000 rows and 2 columns. Each row corresponds to a milestone delay. 
The first column, " Comment" consists of a description of the reason for delay, usually incluing up to three 
sentences. The second column, "Label", is the root cause of the delay based on the comment which can take 
on 11 different categories. These categories are shown in table X. 

## Objective:
The goal is to infere the root cause based on the comments that the superintendents provide schedulers and
stored in the first column. 

## Methodology:
In order to infer the root cause of delays from the comments, X steps were taken: 
### (1) Data Collection:
### (2) Data Cleaning:
### (3) Exploratory Data Analysis (EDA):
### (4) Data Pre-Processing:
 Data was devided into train/test with the 0.6/0.4 ratio. 
### (5) Inference:
 The document was represented using a Bag of Words embbedding method, by tokenizing the words with numbers, counting the frequency of occurance for each token in each document, and normalizing and weighting with diminishing importance tokens that occur in the majority of samples/documents were performed. A corpus of documents was represented by a matrix with one row per document and one column per token occurring in the corpus of documents. Finally, Random Forest algorithm with 1000 estimators and 100 random states were used to train the classifier. 



Results and discussions:
 







