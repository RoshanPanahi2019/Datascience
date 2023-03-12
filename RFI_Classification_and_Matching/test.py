import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity



# returns the cosine similarity value of the two given texts
def compute_cosine_similarity(text1, text2):
    
    # stores text in a list
    list_text = [text1, text2]
    
    # converts text into vectors with the TF-IDF 
    vectorizer = TfidfVectorizer(stop_words='english')
    vectorizer.fit_transform(list_text)
    tfidf_text1, tfidf_text2 = vectorizer.transform([list_text[0]]), vectorizer.transform([list_text[1]])
    
    # computes the cosine similarity
    cs_score = cosine_similarity(tfidf_text1, tfidf_text2)
    print(cs_score)
    return np.round(cs_score[0][0],2)
#======================
if __name__=="__main__":
    sentence1 = 'I like eating ice cream on a hot summer day.'
    sentence2 = 'Only boring people do not like eating ice cream.'
    sentence3 = 'I do not like going out during the summer since it is so hot.'
    compute_cosine_similarity(sentence1,sentence2)