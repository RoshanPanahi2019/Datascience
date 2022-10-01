## Delay root cause inference based on the comment documents, using natural language processing. 

In this project, the comments from schedulers were classified into 11 categories using natural language processing.

## Objective:
The goal is to infere the root cause based on the comments that the superintendents provide schedulers and stored in the first column. 

## Methodology:
In order to infer the root cause of delays from the comments, X steps were taken: 


### 3. Feature Egineering: 
Chi-Square test was used to select the k most relevent features with respect to the label. This helps reduce the sparsity if of the training data, reduce the computational complexity of the model, reduce overfitting, therefore improve the performance of the model with smaller sizes of training data. 

## Case study:

## Description of the dataset:
The dataset consists of a table with 1000 rows and 2 columns. Each row corresponds to a milestone delay. 
The first column, " Comment" consists of a description of the reason for delay, usually incluing up to three 
sentences. The second column, "Label", is the root cause of the delay based on the comment which can take 
on 11 different categories. These categories are shown in table X. 

### 1. Data Collection:
Data was collected from 256 projects, from the company. 
### 2. Data Cleaning:

### 3. Feature Egineering: 
After applying Chi-Squar test, the features with P_value<0.05 and 'nan' were discarded, remaining 205 number of features from the intial 870 features 


### 3. Exploratory Data Analysis (EDA):
Priliminary EDA shows that the data includes 1000 rows and 2 columns, corresponding to the number of projects and the column "Comment" and "Root Cause for Delay" which is treated as labels. Figure below shows the ditribution of the labels. 

![LabelDistribution](https://user-images.githubusercontent.com/55706949/186999007-bf799541-2fef-4cc7-ab29-bb151212daca.png)

As shown in the figure above, the dataset is unbalanced in terms of number of target classes. This would result in bias towards the majority class. Therefore, the X method is used to sample more from the minority class (TBD). 

### 4. Data Pre-Processing:
 Data was devided into train/test with the 0.6/0.4 ratio. 
### 5. Inference:
 The document was represented using a Bag of Words embbedding method, by tokenizing the words with numbers, counting the frequency of occurance for each token in each document, and normalizing and weighting with diminishing importance tokens that occur in the majority of samples/documents were performed. A corpus of documents was represented by a matrix with one row per document and one column per token occurring in the corpus of documents. Finally, Random Forest algorithm with 1000 estimators and 100 random states were used to train the classifier. 



### Results and discussions:
The accuracy of the implemention is %89. 


 ![Comment_Classification_Confusion_Matrix](https://user-images.githubusercontent.com/55706949/186998256-09feca58-8472-4c7e-9e55-1d2277e988e5.png)

 










