# https://medium.com/analytics-vidhya/nlp-tutorial-for-text-classification-in-python-8f19cd17b49e
# https://stackabuse.com/text-classification-with-python-and-scikit-learn/
# Github in VS Code: https://code.visualstudio.com/docs/editor/github
# Random forest --> Read the documentation in https://stackabuse.com/random-forest-algorithm-with-python-and-scikit-learn/

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import ConfusionMatrixDisplay
import matplotlib.pyplot as plt
from pickle import TRUE
from pydoc import doc
from tkinter.tix import COLUMN
import pandas as pd
import re
import nltk
from nltk.stem import WordNetLemmatizer
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
nltk.download('wordnet')
nltk.download('omw-1.4')

def my_read_file(path):
    data=pd.read_csv(path) # read the file to a dataframe. 
    data=data.rename(columns={"Enter Comments ":"Comment","Root_Cause_for_Slip_or_Push":"Label"}) # rename columns.
    return data

documents=[]
stemmer = WordNetLemmatizer()

def pre_process(data):
    for doc in range(len(data)):
        document = re.sub(r'\W', ' ', str(data["Comment"][doc])) # Remove all the special characters
        document = re.sub(r'\s+[a-zA-Z]\s+', ' ', document) # remove all single characters
        document = re.sub(r'\^[a-zA-Z]\s+', ' ', document) # Remove single characters from the start
        document = re.sub(r'\s+', ' ', document, flags=re.I)  # Substituting multiple spaces with single space
        document = re.sub(r'^b\s+', '', document) # Removing prefixed 'b'
        document = document.lower() # Converting to Lowercase
        document = document.split() # Lemmatization
        document = [stemmer.lemmatize(word) for word in document]
        document = ' '.join(document)
        documents.append(document)
    return documents

# Tokenizing & Vectorizing
def vectorize(documents,y):
    vectorizer=CountVectorizer() # Create the vectorizer object. 
    X=vectorizer.fit_transform(documents).toarray() # Tokinize & count number of occurances.
    tfidfconverter = TfidfTransformer() # term frequency–inverse document frequency
    X = tfidfconverter.fit_transform(X).toarray()
    return X

def train_test(X,y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=100) # Split to train & test. 
    classifier = RandomForestClassifier(n_estimators=1000, random_state=100)
    classifier.fit(X_train, y_train)
    y_pred = classifier.predict(X_test)
    print(confusion_matrix(y_test,y_pred))
    print(classification_report(y_test,y_pred))
    print(accuracy_score(y_test, y_pred))
    return(y_test, y_pred)

def cal_accuracy(y_test, y_pred):
        
    print("Confusion Matrix: ", confusion_matrix(y_test, y_pred))
    print ("Accuracy : ", accuracy_score(y_test,y_pred)*100)
    print("Report : ", classification_report(y_test, y_pred))
    ConfusionMatrixDisplay.from_predictions(y_test,y_pred)
    plt.title("Classifying Field Crew Comments to 25 Delay Root Cause Classes")
    plt.xlabel("Predicted Labels")
    plt.ylabel("True Labels")
    plt.show()

# move this to the schedule classifer script. 
#def Visualize(rf):
#    import matplotlib.pyplot as plt
#    from sklearn import tree
#    fn=data.feature_names
#    cn=data.target_names
#    fig, axes = plt.subplots(nrows = 1,ncols = 1,figsize = (4,4), dpi=800)
#    tree.plot_tree(rf.estimators_[0],
#                feature_names = fn, 
#                class_names=cn,
#                filled = True);
#    fig.savefig('rf_individualtree.png')

#==============================================
if __name__=="__main__":
    path="/media/ms/D/myGithub_Classified/Skanska/NLP/Comment_Root_Cause_EntireData_08262022.csv"
    data=my_read_file(path)
    dist_label=data['Label'].value_counts() # Statistics of the dataset.
    data["Label"].hist(bins=100)
    plt.show()
    exit()
    
    documents=pre_process(data)
    vector=vectorize(documents,data["Label"])
    y_test, y_pred=train_test(vector,data["Label"])
    cal_accuracy(y_test, y_pred)