# https://medium.com/analytics-vidhya/nlp-tutorial-for-text-classification-in-python-8f19cd17b49e
# https://stackabuse.com/text-classification-with-python-and-scikit-learn/
# cleaning texts
from pydoc import doc
from tkinter.tix import COLUMN
from nltk.stem.porter import PorterStemmer
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
#for text pre-processing
import re, string
import seaborn as sns
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer
from nltk.corpus import wordnet
from nltk.stem import WordNetLemmatizer
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
nltk.download('wordnet')
nltk.download('omw-1.4')
#for model-building
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report, f1_score, accuracy_score, confusion_matrix
from sklearn.metrics import roc_curve, auc, roc_auc_score

# bag of words
#from sklearn.feature_extraction.text import TfidataVectorizer
from sklearn.feature_extraction.text import CountVectorizer

#for word embedding
import gensim
from gensim.models import Word2Vec

#################################

def my_read_file(path):
    data=pd.read_csv(path) # read the file to a dataframe. 
    data=data.rename(columns={"Enter Comments ":"Comment","Root Cause for Slip / Push":"Label"}) # rename columns.
    return data

# Pre-processing
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

def vectorize(documents,y):
    # Tokenizing & Vectorizing
    from sklearn.feature_extraction.text import CountVectorizer
    from sklearn.feature_extraction.text import TfidfTransformer
    vectorizer=CountVectorizer() # Create the vectorizer object. 
    X=vectorizer.fit_transform(documents).toarray() # Tokinize & count number of occurances.
    #tfidfconverter = TfidfTransformer() # term frequency–inverse document frequency
    #X = tfidfconverter.fit_transform(X).toarray()
    print(X)
    exit()
    return X

def train_test(X,y):
# Training

    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=100) # Split to train & test. 
    # Github in VS Code: https://code.visualstudio.com/docs/editor/github
    # Random forest --> Read the documentation in https://stackabuse.com/random-forest-algorithm-with-python-and-scikit-learn/
    from sklearn.ensemble import RandomForestClassifier

    classifier = RandomForestClassifier(n_estimators=1000, random_state=100)
    classifier.fit(X_train, y_train)
    y_pred = classifier.predict(X_test)

    from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
    print(confusion_matrix(y_test,y_pred))
    print(classification_report(y_test,y_pred))
    print(accuracy_score(y_test, y_pred))
 
# The accuracy is high. Validate the resut by going over the code and the data again. 
# If there are no problems, add in more data, train, and test again. 

#---------------------
if __name__=="__main__":
    path="/media/ms/OS/Users/rosha/Desktop/DS_USB_08012022/Data/NLP/Scope_11039_NLP.csv"
    data=my_read_file(path)
    dist_label=data['Label'].value_counts() # Statistics of the dataset.
    #df["Label"].hist(bins=100)
    #plt.show()
    documents=pre_process(data)
    vector=vectorize(documents,data["Label"])
    train_test(vector,data["Label"])