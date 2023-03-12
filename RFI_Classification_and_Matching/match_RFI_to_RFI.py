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

import string
from operator import itemgetter
import numpy as np

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

def prepare_RFI(In_RFI):     # Read the RFI, pre-pocess it, write it out, read it back into a list of documents. 

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

def my_vectorize(documents,vocab=None): # Tokenizing & Vectorizing
    vectorizer=CountVectorizer(vocabulary=vocab) # Create the vectorizer object. 
    X=vectorizer.fit_transform(documents).toarray() # Tokinize & count number of occurances.
    tfidfconverter = TfidfTransformer() # term frequency–inverse document frequency
    X = tfidfconverter.fit_transform(X).toarray()
    X_names = vectorizer.get_feature_names_out ()
    return np.ceil(X),X_names
    return(np.ceil(X)) # Just return the term frequency for now (If present 1 else 0)

def my_match(query,my_RFI):
    score=[]
    nlp = spacy.load("en_core_web_sm")
    query=nlp(query)
    for i in range(10000):
        rfi=nlp(my_RFI[i])
        score.append(query.similarity(rfi))
    index, element = max(enumerate(score), key=itemgetter(1))
    return (index,element)

#===================================================
if __name__ == "__main__":
    root="/media/ms/D/myGithub_Classified/Skanska/RFI/Data/"
    In_RFI=root+"raw_RFI_data/Input/rfi data 9.05.18 rev2.xlsx"
    In_dwg=root+"OCR_from_drawing/skanska-OST Current Drawings Thru ASI 11 02.28.22/"
    out=root+"output/"

    my_RFI=prepare_RFI(In_RFI)
    raw_RFI=pd.read_excel(In_RFI) # We can use the raw RFI because apparantly no records were droped during pre-processing. 
    raw_RFI=clean(raw_RFI)
    raw_RFI=raw_RFI['question']
    import csv
    with open(out+'match_RFI_to_RFI/'+'RFI_1.txt', 'w') as out_file:
        my_query_vector,f_names=my_vectorize([my_RFI[0]])
        my_RFI_vector,_=my_vectorize(my_RFI,vocab=f_names)

        sim_score=[]
        k=10
        for i in range(len(my_RFI_vector)):
            sim_score.append(int(np.dot(my_query_vector,np.transpose(my_RFI_vector[i]))))
        top_k_match=(sorted(range(len(sim_score)), key=lambda i: sim_score[i], reverse=True)[:k]) #returns the top k best matches 
        for i in range(k): out_file.write((f'{raw_RFI[top_k_match[i]]}\n \n '))

        from collections import Counter
        dict=Counter(sorted(sim_score))

        list=sorted(dict.items())
        x,y=zip(*list)
        x_percent=[x2/max(x) for x2 in x ]
        import matplotlib.pylab as plt
        plt.plot(x_percent,y)
        plt.title("Matching Score for RFIs", size=15)
        plt.xlabel('matching score %')
        plt.ylabel('number of RFIs')

        plt.show()
    out_file.close()
    exit()












    # Trying matching based on presence of the tags ## Continue from here: We just matched based on presence of these terms. The result is not very relevent. Lets check out first 10 results. Then add more RFIs. Then think howelse we can return more relevent results..
    my_hospital_equipment_tags=['ultrasound','xray','nurse']
    my_critial_tags=['embed']
    my_structural_tags=['column','beam','slab','exposed steel']
    my_fire_tags=['fire','alarm','extinguisher']
    my_electrical_tags=['receptacle','electrical','thermostat','panel','wire','cable','conduit','electrician']
    my_arch_tags=['architect','toilet','window','wheelchair','sink','soap dispenser','towel dispenser']
    my_mechnical_tags=['mechanical','plumbing','piping']
    my_custom_tags=['cost','embed','change order']
    my_all_tag=my_hospital_equipment_tags+my_critial_tags+my_structural_tags+my_fire_tags+my_electrical_tags+my_arch_tags+my_mechnical_tags
    # Find the most related RFI to the current drawing based on these tags. 

        #print(sum(my_RFI_vector[i]))
    #index, element = max(enumerate(sim_score), key=itemgetter(1))
    #[print(i) for i, j in enumerate(element) if j > (element[0]-5)]
    #print(sum(my_RFI_vector[i]).sort())
    print("\n============== query ============\n") 
    print(index, element)
    print("\n")
    print(query)
    print("\n============== RFI ============\n")
    print(my_RFI[index])
    #Jaccart Distance(my_query_vector,my_RFI_vector[0])
    exit()
    print(len(my_RFI_vector[0]),my_terms)
    exit()


    # Try matching using semantics
    index,score=my_match(query, my_RFI)
    print('\n================== Query =====================\n')
    print(index, score)
    print('\n')
    print(query)
    print('\n================== Match =====================\n')
    print(my_RFI[index])
  
  # Result was not good:
    # Example dwg was matched with irelevant RFIs
    # Solution:
        # Use matching based on occurance of exact words
        # clean the dwg file from nonsense
        # Filter the RFI and DWG on domain words and do the matchin again
        # Match based on tagging. 
