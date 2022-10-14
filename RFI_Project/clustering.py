#from tkinter.tix import COLUMN
import pandas as pd

# A better wat maybe to use the list of keywords as training data, and classify the records to disciplines accordingly. 

# create a tag column and for each record append tags to a list of tags. 
# create a discipline column and for each record append discipline to a list of disciplines. 

def clean(df): 
       list_drop=["hour","date","cost impact","schedule impact","drawing impact",'Priority', 'Merged Category', 'reason',
              'job name', 'job number', 'start date', 'end date', 'Contract value']
       df.drop(list_drop,axis="columns",inplace=True)
       df.insert(df.shape[1],"discipline","") # insert a new column, and initialze with empty string value. 
       df.dropna(subset=['question'],axis=0,inplace=True) # drop rows having 'question column with NA value. 
       return(df)
       
def cluster(df,keyword):
       keyword_row_index=df[df['question'].str.contains('|'.join(keyword))].index # from 'question' column, return the index of the rows containg the keyword
       df["discipline"].loc[keyword_row_index.tolist()]="Fire_Protection" # access the indexed rows in the displine column and update their value. 
       print( df["discipline"].loc[keyword_row_index.tolist()])
       return (df)

#==============
if __name__ == "__main__":
       data_path="/media/ms/D/myGithub_Classified/Skanska/Data/rfi-data-dump/Input/rfi data 9.05.18 rev2.xlsx"
       keywords_path="/media/ms/D/myGithub_Classified/Skanska/Data/rfi-data-dump/Annotation/keywords.csv"
       df=pd.read_excel(data_path)
       df=clean(df)
       keywords=pd.read_csv(keywords_path)
       # Do this for all "discplines" in the keword dataframe
       keywords=keywords["Fire_Protection"].dropna().tolist()
       cluster(df,keywords)
