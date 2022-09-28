# https://medium.com/analytics-vidhya/nlp-tutorial-for-text-classification-in-python-8f19cd17b49e
# https://stackabuse.com/text-classification-with-python-and-scikit-learn/
# Github in VS Code: https://code.visualstudio.com/docs/editor/github
# Random forest --> Read the documentation in https://stackabuse.com/random-forest-algorithm-with-python-and-scikit-learn/
# Word to Vec: https://towardsdatascience.com/text-classification-with-nlp-tf-idf-vs-word2vec-vs-bert-41ff868d1794
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from nltk.stem import WordNetLemmatizer
from sklearn.model_selection import RandomizedSearchCV# Number of trees in random forest
from pprint import pprint
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import ConfusionMatrixDisplay
from pickle import TRUE
from tkinter.tix import COLUMN
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import re
import nltk
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
nltk.download('wordnet')
nltk.download('omw-1.4')

def my_read_file(path):
    data=pd.read_csv(path) # read the file to a dataframe. 
    data=data.rename(columns={"Enter Comments ":"Comment","Root_Cause_for_Slip_or_Push":"Label"}) # rename columns.
    return data

def EDA(data):
    dist_label=data['Label'].value_counts() # Statistics of the dataset.
    data["Label"].hist(bins=100)
    plt.show()
    return(0)

def pre_process(data):
    documents=[]
    stemmer = WordNetLemmatizer()
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

def vectorize(documents,y): # Tokenizing & Vectorizing
    vectorizer=CountVectorizer() # Create the vectorizer object. 
    X=vectorizer.fit_transform(documents).toarray() # Tokinize & count number of occurances.
    tfidfconverter = TfidfTransformer() # term frequency–inverse document frequency
    X = tfidfconverter.fit_transform(X).toarray()
    return X

# Working on this code-block./ done:learn the math / learn the implementation.
def word2vec():
  class MeanEmbeddingVectorizer(object):
    def __init__(self, word2vec):
        self.word2vec = word2vec
        # if a text is empty we should return a vector of zeros
        # with the same dimensionality as all the other vectors
        self.dim = len(next(iter(word2vec.values())))  
    def fit(self, X, y):
            return self

    def transform(self, X):
            return np.array([
                np.mean([self.word2vec[w] for w in words if w in self.word2vec]
                        or [np.zeros(self.dim)], axis=0)
                for words in X
            ])

    w2v = dict(zip(model.wv.index2word, model.wv.syn0)) 
    df['clean_text_tok']=[nltk.word_tokenize(i) for i in df['clean_text']]
    model = Word2Vec(df['clean_text_tok'],min_count=1)     
    modelw = MeanEmbeddingVectorizer(w2v)

    # converting text to numerical data using Word2Vec
    X_train_vectors_w2v = modelw.transform(X_train_tok)
    X_val_vectors_w2v = modelw.transform(X_test_tok)

def grid_search():
    n_estimators = [int(x) for x in np.linspace(start = 200, stop = 2000, num = 10)] # Number of features to consider at every split
    max_features = ['auto', 'sqrt'] # Maximum number of levels in tree
    max_depth = [int(x) for x in np.linspace(10, 110, num = 11)]
    max_depth.append(None)
    min_samples_split = [2, 5, 10]     # Minimum number of samples required to split a node
    min_samples_leaf = [1, 2, 4]     # Minimum number of samples required at each leaf node
    bootstrap = [True, False] # Method of selecting samples for training each tree
    random_grid = {'n_estimators': n_estimators, # Create the random grid
                'max_features': max_features,
                'max_depth': max_depth,
                'min_samples_split': min_samples_split,
                'min_samples_leaf': min_samples_leaf,
                'bootstrap': bootstrap}
    return(random_grid)

def train(X,y,param_grid):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=100) # Split to train & test. 
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
    #plt.show()

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
    #EDA(data)
    documents=pre_process(data)
    vector=vectorize(documents,data["Label"])
    param_grid=grid_search()
    X_test,y_test,rf_model_best=train(vector,data["Label"],param_grid)
    y_pred = evaluate(rf_model_best, X_test, y_test)
    cal_accuracy(y_test, y_pred) 