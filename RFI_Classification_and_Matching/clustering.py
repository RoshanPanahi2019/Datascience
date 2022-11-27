#from tkinter.tix import COLUMN
from ast import And
import pandas as pd

# A better way maybe to use the list of keywords as training data, and classify the records to disciplines accordingly. 

# create a tag column and for each record append tags to a list of tags. 
# create a discipline column and for each record append discipline to a list of disciplines. 

def clean(df): 
       list_drop=["hour","date","cost impact","schedule impact","drawing impact",'Priority', 'Merged Category', 'reason',
              'job name', 'job number', 'start date', 'end date', 'Contract value']
       df.drop(list_drop,axis="columns",inplace=True)
       df.insert(df.shape[1],"discipline","") # insert a new column, and initialze with empty string value. 
       df.dropna(subset=['question'],axis=0,inplace=True) # drop rows having 'question column with NA value. 
       return(df)
       
def cluster(df,key_for_disciplines):
       for discipline in key_for_disciplines:
              keywords=key_for_disciplines[discipline].dropna().tolist()
              if keywords==[]: 
                     continue
              keyword_row_index=((df[df['question'].str.contains('|'.join(keywords))]["discipline"]=="").index) # from 'question' column, return the index of the rows containg the keyword AND they have not yet been clustered              
              df["discipline"].loc[keyword_row_index.tolist()]=discipline # access the indexed rows in the displine column and update their value. 
       return (df)

#==============
if __name__ == "__main__":
       root_dir="/media/ms/D/myGithub_Classified/Skanska/Data/rfi-data-dump/"
       data_path=root_dir+"Input/rfi data 9.05.18 rev2.xlsx"
       keywords_path=root_dir+"Annotation/keywords.csv"
       output_path=root_dir+"output/"
       df=pd.read_excel(data_path)
       df=clean(df)
       key_for_disciplines=pd.read_csv(keywords_path)
       df_clustered=cluster(df,key_for_disciplines)
       df_clustered.to_excel(output_path+"RFI_Clustered_by_Descipline.xlsx")

       # see how many rows have been clustered
       # fix the spelling error
       # increase number of keywords
              # Use keywords with 100% confidence. 
              # Repeat until good amount is clustered. 
              # Think of the pipeline again. 
              # Next step: 
                     # 