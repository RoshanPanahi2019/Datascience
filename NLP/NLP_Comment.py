# cleaning texts
from tkinter.tix import COLUMN
import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import CountVectorizer
 
path_input="/media/ms/OS/Users/rosha/Desktop/DS_USB_07302022/Data/NLP/Scope_11039_NLP.csv"
df=pd.read_csv(path_input)
df=df.rename(columns={"Enter Comments ":"Comment","Root Cause for Slip / Push":"Label"})       

dt=nltk.download('stopwords')
corpus = []
# Continue with preprocessing the documents.  



for i in range(0, 5):
    text = re.sub('[^a-zA-Z]', '', dataset['Text'][i])
    text = text.lower()
    text = text.split()
    ps = PorterStemmer()
    text = ''.join(text)
    corpus.append(text)
 
# creating bag of words model
cv = CountVectorizer(max_features = 1500)
 
X = cv.fit_transform(corpus).toarray()
y = dataset.iloc[:, 1].values