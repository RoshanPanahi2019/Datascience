#
# https://www.kaggle.com/code/roshanposu/a-simple-lstm-based-time-series-classifier/edit
# https://towardsdatascience.com/building-rnn-lstm-and-gru-for-time-series-using-pytorch-a46e5b094e7b

# imports
from cProfile import label
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import confusion_matrix
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.ensemble import RandomForestClassifier

# Function to perform training with giniIndex.
def train_using_gini(X_train, X_test, y_train):
  
    # Creating the classifier object
    clf_gini = DecisionTreeClassifier(criterion = "gini",
            random_state = 100,max_depth=3, min_samples_leaf=5)
  
    # Performing training
    clf_gini.fit(X_train, y_train)
    return clf_gini

# Function to perform training with entropy.
def train_using_entropy(X_train, X_test, y_train):
  
    # Decision tree with entropy
    clf_entropy = DecisionTreeClassifier(
            criterion = "entropy", random_state = 100,
            max_depth = 3, min_samples_leaf = 5)
  
    # Performing training
    clf_entropy.fit(X_train, y_train)
    return clf_entropy

def train_random_forest(X_train, X_test,y_train,y_test):
    classifier = RandomForestClassifier(n_estimators=1000, random_state=100)
    classifier.fit(X_train, y_train)   
    y_pred = classifier.predict(X_test)
    print(confusion_matrix(y_test,y_pred))
    print(classification_report(y_test,y_pred))
    print(accuracy_score(y_test, y_pred))   


# Function to make predictions
def prediction(X_test, clf_object):
  
    # Predicton on test with giniIndex
    y_pred = clf_object.predict(X_test)
    print("Predicted values:")
    print(y_pred)
    return y_pred
      
# Function to calculate accuracy
def cal_accuracy(y_test, y_pred):
      
    print("Confusion Matrix: ", confusion_matrix(y_test, y_pred))
    print ("Accuracy : ", accuracy_score(y_test,y_pred)*100)
    print("Report : ", classification_report(y_test, y_pred))
    ConfusionMatrixDisplay.from_predictions(y_test,y_pred)
    plt.title("Classification of Projects to High/Low Risk \n Based on Observed Reasons for Delay")
    plt.xlabel("Predicted Labels")
    plt.ylabel("True Labels")
    plt.show()

def main():
    #Pre-processing 
    path="/media/ms/D/myGithub_Classified/Skanska/Data_Source/Out/Merge1_Tbl_1_2_4_DelayAmount_Reduced_Freq_Reduced_Label.csv"
    path="/media/ms/D/myGithub_Classified/Skanska/Data_Source/Out/Merge_Tbl_1_3_Freq_Reduced_Label_OneHotEncoded.csv"
    df=pd.read_csv(path)
    row,column=df.shape

    ## split dataset %70/%30 
    X = df.values[:, 0:column-2]
    Y = df.values[:,column-1]
    
    from sklearn.preprocessing import StandardScaler
    #sc = StandardScaler()
   

    X_train, X_test, y_train, y_test = train_test_split( 
            X, Y, test_size = 0.3, random_state = 100)

    #X_train = sc.fit_transform(X_train)
    #X_test = sc.transform(X_test)


    print(" Random forest-------->")
    train_random_forest(X_train, X_test, y_train,y_test)
    print(" Decision tree -------->")
    clf_gini = train_using_gini(X_train, X_test, y_train)
    clf_entropy = train_using_entropy(X_train, X_test, y_train)
    

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