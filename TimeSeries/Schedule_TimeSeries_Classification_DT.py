#
# https://www.kaggle.com/code/roshanposu/a-simple-lstm-based-time-series-classifier/edit
# https://towardsdatascience.com/building-rnn-lstm-and-gru-for-time-series-using-pytorch-a46e5b094e7b

from logging import RootLogger
from multiprocessing import cpu_count
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import TensorDataset, DataLoader
from torch.optim.lr_scheduler import _LRScheduler
from sklearn.metrics import confusion_matrix
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report


# Function to perform training with giniIndex.
def train_using_gini(X_train, X_test, y_train):
  
    # Creating the classifier object
    clf_gini = DecisionTreeClassifier(criterion = "gini",
            random_state = 100,max_depth=3, min_samples_leaf=5)
  
    # Performing training
    clf_gini.fit(X_train, y_train)
    return clf_gini

# Function to perform training with entropy.
def tarin_using_entropy(X_train, X_test, y_train):
  
    # Decision tree with entropy
    clf_entropy = DecisionTreeClassifier(
            criterion = "entropy", random_state = 100,
            max_depth = 3, min_samples_leaf = 5)
  
    # Performing training
    clf_entropy.fit(X_train, y_train)
    return clf_entropy

# Function to make predictions
def prediction(X_test, clf_object):
  
    # Predicton on test with giniIndex
    y_pred = clf_object.predict(X_test)
    print("Predicted values:")
    print(y_pred)
    return y_pred
      
# Function to calculate accuracy
def cal_accuracy(y_test, y_pred):
      
    print("Confusion Matrix: ",
        confusion_matrix(y_test, y_pred))
      
    print ("Accuracy : ",
    accuracy_score(y_test,y_pred)*100)
      
    print("Report : ",
    classification_report(y_test, y_pred))

def main():
    # Pre-processing
    path="/media/ms/OS/Users/rosha/Desktop/DS_USB_08012022/Data/Skanksa_Schedule_Classification_HotEncoded_08092022.csv"
    df=pd.read_csv(path)
    df=df.drop(columns="Scope")
    row,column=df.shape
    for clm in range(column):
        new="F"+str(clm)
        df = df.rename(columns={df.columns[clm]: new})

    # One-hot encoding (Why the umber of rows less and more than median are not equal?)
    F27_median=df["F27"].median()
    for rw in range(row):
        if df["F27"][rw]<=F27_median:
            df["F27"][rw]=0
        else:
            df["F27"][rw]=1
    df["F27"].plot.hist(bins=100, alpha=0.5)

    X = df.values[:, 0:column-2]
    Y = df.values[:,column-1]

    # split dataset %70/%30 
    X_train, X_test, y_train, y_test = train_test_split( 
            X, Y, test_size = 0.3, random_state = 100)

    clf_gini = train_using_gini(X_train, X_test, y_train)
    clf_entropy = tarin_using_entropy(X_train, X_test, y_train)
      
    print("Results Using Gini Index:")
      
    # Prediction using gini
    y_pred_gini = prediction(X_test, clf_gini)
    cal_accuracy(y_test, y_pred_gini)
      
    print("Results Using Entropy:")
    # Prediction using entropy
    y_pred_entropy = prediction(X_test, clf_entropy)
    cal_accuracy(y_test, y_pred_entropy)
      
      
# ==================================
if __name__=="__main__":
    main()