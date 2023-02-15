from sklearn.feature_extraction.text import TfidfVectorizer

#documents = [open(f).read() for f in text_files]
documents = 'Hello hi there!'
tfidf = TfidfVectorizer().fit_transform(documents)
print(tfidf)
exit()
# no need to normalize, since Vectorizer will return normalized tf-idf
pairwise_similarity = tfidf * tfidf.T








exit()
import spacy

nlp = spacy.load('en_core_web_sm')
doc1 = nlp(u'Hello hi there!')
doc2 = nlp(u'Hello hi there!')
doc3 = nlp(u'Hey whatsup?')

print (doc1.similarity(doc3)) # 0.999999954642
#print (doc2.similarity(doc3)) # 0.699032527716
#print (doc1.similarity(doc3)) # 0.699032527716
