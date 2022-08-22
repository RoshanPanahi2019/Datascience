# imports
from os import read
from time import process_time_ns
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split

def my_read_file(path):
    data=pd.read_csv(path)
    data=data.drop(columns="Scope")
    #data=data.drop(columns="F26_freq:Scope/Schedule Development During Rev-1")
    #data=data.drop(columns="F0_freq:DWN (Impact to OA)")
    data=data.abs()

    return data

def my_one_hot_encode(data, column_name): # One-Hot-Encoding the label
    m,n=data.shape
    median=data[column_name].median()
    for rw in range(m):
        if data[column_name][rw]<=median:
            data[column_name][rw]=0
        else:
            data[column_name][rw]=1
    data.to_csv("/media/ms/D/myGithub_Classified/Skanska/Data_Source/Out/Merge_Tbl_1_3_Freq_Reduced_Label_OneHotEncoded.csv", index=False )
    return data

def my_normalize(data):
    #print(data.head(5))
    return data
    
#----------------------------
if __name__=="__main__":
    path="/media/ms/D/myGithub_Classified/Skanska/Data_Source/Merge1_Tbl_1_2_4_DelayAmount_Reduced_Freq_Reduced_Label.csv"
    path="/media/ms/D/myGithub_Classified/Skanska/Data_Source/Merge_Tbl_1_3_Freq_Reduced_Label.csv"
    data=my_read_file(path) 
    data=my_one_hot_encode(data,"Target_Sum_Delay_SignOff")
    #my_normalize(data)