import spacy
import os
from ast import And
import pandas as pd
import re
import nltk as nltk
from sklearn.feature_extraction.text import CountVectorizer,TfidfTransformer
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn.metrics.pairwise import cosine_similarity

import shutil
import string
from operator import itemgetter
import numpy as np
import csv

nltk.download('stopwords')

def clean(df): 

    list_drop=["hour","date","cost impact","schedule impact","drawing impact",'Priority', 'Merged Category', 'reason',
            'job name', 'job number', 'start date', 'end date', 'Contract value']
    df.drop(list_drop,axis="columns",inplace=True)
    df.insert(df.shape[1],"discipline","") # insert a new column, and initialze with empty string value. 
    df.dropna(subset=['question'],axis=0,inplace=True) # drop rows having 'question column with NA value. 
    return(df)

# Recrod 10888 was manually deleted since it was blank.
## We are using half the data. Handle bugs and try to pre-process (clean) the entire data. 
def pre_process(df): 
    stemmer = WordNetLemmatizer()
    #df= df.dropna(axis="rows") # drop all records containing Null in the 'question' column
    df= df.apply(lambda x: re.sub('[%s]' % re.escape(string.punctuation), '' , x)) # remove punctuations from the entire text file. 
 
    for doc in range(len(df)): # This is slow for large files. Do more research for a faster approach.. ## Handle records that are equations and start with "=" ### Remove single characteres #### Remove the records that have blank 'question' ##### Review for a better cleaning. 
        document= re.sub(r'[0-9]', '', df[doc]) # Remove all numberic values. 
        document= re.sub("\s+", " ", re.sub( r"((?<=^)|(?<= )).((?=$)|(?= ))", '', document).strip()) #Remove all single charachters. 
        document = re.sub(r'\W', ' ', str(document)) # Remove all the special characters
        document = re.sub(r'\s+', ' ', document, flags=re.I)  # Substituting multiple spaces with single space
        document = document.lower() # Converting to Lowercase
        document = document.split() # Lemmatization
        document = [stemmer.lemmatize(word) for word in document]
        document=[word for word in document if not word in stopwords.words()]
        document = ' '.join(document)
    return (document)

def prepare_RFI(In_RFI,out):     # Read the RFI, pre-pocess it, write it out, read it back into a list of documents. 
    if not os.path.exists(out+'pre_processed/RFI/pre_processed.txt'):
            print("I'm preprocessing the RFI...")
            df_RFI=pd.read_excel(In_RFI)
            df_RFI=clean(df_RFI)
            document=pre_process(df_RFI['question']) # Make preprocessing faster. 
            write(document,out+'pre_processed/RFI/pre_processed.txt')
    else: print("Already exists: RFI preprocessed file")
    my_RFI=read_to_list(out+'pre_processed/RFI/pre_processed.txt')
    return(my_RFI)

def read_to_list(document):
    documents=[]
    with open(document, 'r') as filehandle:
        print("I found the preprocessed file!")
        for line in filehandle:
          curr_place = line[:-1] # Remove linebreak which is the last character of the string
          documents.append(curr_place)
    filehandle.close
    return (documents)

def prepare_dwg(In_dwg):     # Read the DWG, pre-pocess it, write it out, read it back into a list of documents. 
    df_dwg=pd.read_csv(In_dwg)
    df_dwg.drop(['Unnamed: 0'], axis=1, inplace=True)
    my_string=' '.join(map(str,df_dwg['keywords'].tolist()))
    df_dwg=pd.DataFrame({'doc':[my_string]})
    df_dwg['doc']=pre_process(df_dwg['doc'])
    query=df_dwg['doc'] # query is a string of the extract texts from dwg. 
    return(query)

def write(document,my_out):
    with open(my_out, 'w') as out_file:
        out_file.write(f'{document}\n')
        out_file.close()
    return

def my_vectorize(documents,vocab): # Tokenizing & Vectorizing
    vectorizer=CountVectorizer(vocabulary=vocab) # Create the vectorizer object. 
    X=vectorizer.fit_transform(documents).toarray() # Tokinize & count number of occurances.
    tfidfconverter = TfidfTransformer() # term frequency–inverse document frequency
    X = tfidfconverter.fit_transform(X).toarray()
    X_names = vectorizer.get_feature_names_out ()
    #return (np.ceil(X),X_names) # Just return the term frequency for now (If present 1 else 0)
    return(X,X_names) 

def my_match_simple(In_dwg,out,my_RFI,raw_RFI): # match the query dwg with RFI only based on the terms in common. 
    for file in os.listdir(In_dwg):
        match=[]
        if file[-3:]=="csv":
            with open(out+'match/simple/'+file[:-4]+'.csv', 'w') as out_file:
                writer=csv.writer(out_file)
                writer.writerow(["Index","Subject","Question","Answer","Keywords"])
                my_query=prepare_dwg(In_dwg+file)
                my_query_vector,f_names=my_vectorize(my_query,vocab=None)
                my_RFI_vector,_=my_vectorize(my_RFI,vocab=f_names)

                sim_score=[]
                k=10
                for i in range(len(my_RFI_vector)):
                    sim_score.append(int(np.dot(my_query_vector,np.transpose(my_RFI_vector[i]))))

                top_k_match=(sorted(range(len(sim_score)), key=lambda i: sim_score[i], reverse=True)[:k]) #returns the top k best matches 
                for i in range(k): 
                    #out_file.write((f'{raw_RFI[top_k_match[i]]}\n \n '))
                    keys=[]
                    for j in range(len(f_names)): # write the matched keys.
                        if my_RFI_vector[top_k_match][i][j]==1: 
                            keys.append(f_names[j])
                    writer.writerow([i,raw_RFI['subject'][top_k_match[i]],raw_RFI['question'][top_k_match[i]],raw_RFI['answer '][top_k_match[i]],keys])
                keys=[]
    return 0

def my_match_TFITF(raw_RFI,my_RFI,in_root_tmp): #match query with RFI using TFITF

    with open (in_root_tmp+'corpus.txt') as file:
        f=csv.reader(file)
        os.remove(in_root_tmp+'corpus.txt')
        my_query_list=list(f)[0]

        my_query_vector,f_names=my_vectorize(my_query_list,vocab=None)
        my_RFI_vector,_=my_vectorize(my_RFI,vocab=f_names)

        for i in range(len(my_query_vector)):
            sim_score=[]
            with open(in_root_tmp+str(i+1)+'.csv', 'w') as out_file:
                writer=csv.writer(out_file)
                for j in range(len(my_RFI_vector)):
                    sim_score.append(np.dot(my_query_vector[i],np.transpose(my_RFI_vector[j])))
                k=10
                top_k_match=(sorted(range(len(sim_score)), key=lambda i: sim_score[i], reverse=True)[:k]) #returns the top k best matches 
                
                for m in range(k): 
                    #out_file.write((f'{raw_RFI[top_k_match[i]]}\n \n '))
                    keys=[]
                    for j in range(len(f_names)): # write the matched keys.
                        if my_RFI_vector[top_k_match][m][j]>0 and my_query_vector[i][j]>0: 
                            keys.append(f_names[j])
                    writer.writerow([m,raw_RFI['subject'][top_k_match[m]],raw_RFI['question'][top_k_match[m]],raw_RFI['answer '][top_k_match[m]],keys])
                keys=[]
    return (0)

def my_match_spacy(query,my_RFI):
    score=[]
    nlp = spacy.load("en_core_web_sm")
    query=nlp(query)
    for i in range(10000):
        rfi=nlp(my_RFI[i])
        score.append(query.similarity(rfi))
    index, element = max(enumerate(score), key=itemgetter(1))
    return (index,element)

# Store all texts from a drawing in a list file. len(list_file)=[number of sheets]
def csv_to_list(in_root_tmp):
    print("Creating corpus list...")
    my_query_list=[]
    for file in sorted(os.listdir(in_root_tmp)):
        my_query=prepare_dwg(in_root_tmp+file) # Still need to remove none sense
        my_query_list.append(my_query.to_list())
        os.remove(in_root_tmp+file)
    write(my_query_list,in_root_tmp+"corpus.txt")

def run():
    root="/media/ms/D/myGithub_Classified/Skanska/RFI/Data/"
    In_RFI=root+"raw_RFI_data/Input/rfi data 9.05.18 rev2.xlsx"
    out="/media/ms/D/myGithub/Datascience/RFI_Classification_and_Matching/Website/static/out/match/"
    in_root_tmp="/media/ms/D/myGithub/Datascience/RFI_Classification_and_Matching/tmp/"

    csv_to_list(in_root_tmp)
    my_RFI=prepare_RFI(In_RFI,out)
    raw_RFI=pd.read_excel(In_RFI) # We can use the raw RFI because apparantly no records were droped during pre-processing. 
    raw_RFI=clean(raw_RFI)
    my_match_TFITF(raw_RFI,my_RFI,in_root_tmp)   


