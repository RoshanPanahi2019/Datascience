## Delay root cause inference based on the comment documents, using natural language processing. 

In this project, the comments from schedulers were classified into 11 categories using natural language processing.

##Description of the dataset:
The dataset consists of a table with 71 rows and 2 columns. Each row corresponds to a milestone delay. 
The first column, " Comment" consists of a description of the reason for delay, usually incluing up to three 
sentences. The second column, "Label", is the root cause of the delay based on the comment which can take 
on 11 different categories. These categories are shown in table X. 

##Objective:
The goal is to infere the root cause based on the comments that the superintendents provide schedulers and
stored in the first column. 

##Methodology:
To make the inference, natural language processing is used. 

### Exploratory data analysis (EDA):

### Data processing:
Bag of words (word embbedding) representation method was used to represent the documents with numbers as features. 

#### Tokenizing: words were treated as tokens and given an integer id for each possible token.
#### Counting: occurrences of tokens were counted in each document.
#### Normalizing: normalizing and weighting with diminishing importance tokens that occur in the majority of samples/documents were performed.
(More specific definition)
A corpus of documents was represented by a matrix with one row per document and one column per token occurring in the corpus of documents.

Results and discussions:
 







