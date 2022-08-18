# imports
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split

def pre_process_1():
    #Pre-processing 
    path="/media/ms/D/myGithub_Classified/Skanska/Schedule_Pred_DelayTypeFreq_CC_Delay.csv"
    df=pd.read_csv(path)
 
    df=df.drop(columns="Scope")
    row,column=df.shape

    for clm in range(column):
        new="F"+str(clm)
        df = df.rename(columns={df.columns[clm]: new})
    df=abs(df)
    df=df.rename(columns={"F26":"Label"})

    # Data structure: df={Scope, Features[Fequency], Label[0/1]}

    ## One-Hot-Encoding 
    Label_median=df["Label"].median()

    for rw in range(row):
        ### One-Hot-Encoding the label
        if df["Label"][rw]<=Label_median:
            df["Label"][rw]=0
        else:
            df["Label"][rw]=1
            
        ### One-Hot-Encoding the features-Frequency ignored.
        #for cl in range (clm-1):
        #    if df.loc[rw][cl]>0:
        #        df.loc[rw][cl]=1
        #   else:
        #       df.loc[rw][cl]=0

    #df["Label"].plot.hist(bins=100, alpha=0.5)

    df.to_csv("/media/ms/D/myGithub_Classified/Skanska/Schedule_Pred_DelayTypeFreq_CC_Delay_OneHotEncoded.csv", index=False )

####################################
if __name__=="__main__":
    pre_process_1()