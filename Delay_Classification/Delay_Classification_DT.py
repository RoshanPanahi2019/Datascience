#
# https://www.kaggle.com/code/roshanposu/a-simple-lstm-based-time-series-classifier/edit
# https://towardsdatascience.com/building-rnn-lstm-and-gru-for-time-series-using-pytorch-a46e5b094e7b

# imports
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from cProfile import label
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import confusion_matrix
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV# Number of trees in random forest
from pprint import pprint

def my_read_file(path):
    data=pd.read_csv(path)
    row,column=data.shape
    X = data.values[:, 0:column-2] 
    y = data.values[:,column-1]
    return X,y

def train_using_gini(X_train, X_test, y_train): # Function to perform training with giniIndex.
    clf_gini = DecisionTreeClassifier(criterion = "gini",
            random_state = 100,max_depth=3, min_samples_leaf=5) # Creating the classifier object
    clf_gini.fit(X_train, y_train)   # Performing training
    return clf_gini

def train_using_entropy(X_train, X_test, y_train): # Function to perform training with entropy.
    clf_entropy = DecisionTreeClassifier(
            criterion = "entropy", random_state = 100,
            max_depth = 3, min_samples_leaf = 5) # Decision tree with entropy
    clf_entropy.fit(X_train, y_train) # Performing training
    return clf_entropy

def grid_search():
    n_estimators = [int(x) for x in np.linspace(start = 800, stop = 2000, num = 10)] # Number of features to consider at every split
    max_features = ['auto', 'sqrt'] # Maximum number of levels in tree
    max_depth = [int(x) for x in np.linspace(10, 110, num = 11)]
    max_depth.append(None)
    min_samples_split = [2, 5, 10]     # Minimum number of samples required to split a node
    min_samples_leaf = [1, 2, 4]     # Minimum number of samples required at each leaf node
    bootstrap = [True, False] # Method of selecting samples for training each tree
    random_grid = {#'n_estimators': n_estimators, # Create the random grid
                #'max_features': max_features,
                #'max_depth': None,
               # 'min_samples_split': min_samples_split,
                #'min_samples_leaf': min_samples_leaf,
                #'bootstrap': bootstrap
                }
    return(random_grid)

def train(X,y,param_grid):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=100) # Split to train & test. 
    rf = RandomForestClassifier(n_estimators=1000, random_state=100)
    rf_random = RandomizedSearchCV(estimator = rf, param_distributions = param_grid, n_iter = 100, cv = 3, verbose=2, random_state=42, n_jobs = -1)
    rf_random.fit(X_train, y_train)
    print(rf_random.best_params_)
    return(X_test,y_test,rf_random.best_estimator_)

def evaluate(model, X_test, y_test):
    predictions = model.predict(X_test)
    return predictions

def cal_accuracy(y_test, y_pred):
        
    print("Confusion Matrix: ", confusion_matrix(y_test, y_pred))
    print ("Accuracy : ", accuracy_score(y_test,y_pred)*100)
    print("Report : ", classification_report(y_test, y_pred))
    ConfusionMatrixDisplay.from_predictions(y_test,y_pred)
    plt.title("Classifying Field Crew Comments to 25 Delay Root Cause Classes")
    plt.xlabel("Predicted Labels")
    plt.ylabel("True Labels")

def prediction(X_test, clf_object): # Function to make predictions
    y_pred = clf_object.predict(X_test) # Predicton on test with giniIndex
    print("Predicted values:")
    print(y_pred)
    return y_pred
      
def cal_accuracy(y_test, y_pred): # Function to calculate accuracy
    print("Confusion Matrix: ", confusion_matrix(y_test, y_pred))
    print ("Accuracy : ", accuracy_score(y_test,y_pred)*100)
    print("Report : ", classification_report(y_test, y_pred))
    ConfusionMatrixDisplay.from_predictions(y_test,y_pred)
    plt.title("Classification of Projects to High/Low Risk \n Based on Observed Reasons for Delay")
    plt.xlabel("Predicted Labels")
    plt.ylabel("True Labels")
    plt.show()

def other_methods():
        print(" Decision tree -------->")
        clf_gini = train_using_gini(X_train, X_test, y_train)
        clf_entropy = train_using_entropy(X_train, X_test, y_train)
        print("Results Using Gini Index:")
        y_pred_gini = prediction(X_test, clf_gini) # Prediction using gini
        cal_accuracy(y_test, y_pred_gini)
        print("Results Using Entropy:")
        y_pred_entropy = prediction(X_test, clf_entropy)    # Prediction using entropy
        cal_accuracy(y_test, y_pred_entropy)
      
# ==================================
if __name__=="__main__":
    path="/media/ms/D/myGithub_Classified/Skanska/Data_Source/Out/Merge_Tbl_1_3_Freq_Reduced_Amount_Reduced_Budget_Label_OneHotEncoded.csv"
    X,y=my_read_file(path)
    param_grid=grid_search()
    X_test,y_test,rf_model_best=train(X,y,param_grid)
    
    y_pred = evaluate(rf_model_best, X_test, y_test)
    cal_accuracy(y_test, y_pred) 
    exit()




    train_random_forest(X_train, X_test, y_train,y_test)