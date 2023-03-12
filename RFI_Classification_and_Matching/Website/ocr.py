# This program takes pdf drawings and outputs the texts for each sheet. 

import os
from os import listdir, mkdir
import pytesseract
from pytesseract import Output
from pdf2image import convert_from_path
from PIL import Image
import pandas as pd
import shutil

def run():
    #TODO: make all the produced files, temporary files, to increase access speed. 
    In_root_tmp='/media/ms/D/myGithub/Datascience/RFI_Classification_and_Matching/tmp/'

    for file in listdir(In_root_tmp):
        convert_from_path(In_root_tmp+file,fmt='jpg',output_folder=In_root_tmp)

    for file in listdir(In_root_tmp):
        if file[-3:]=="pdf":
            os.remove(In_root_tmp+file)

    for file in sorted(listdir(In_root_tmp)):
        if file[-3:]=="jpg":
            img = Image.open(In_root_tmp+file)
            data = pytesseract.image_to_data(img, output_type=Output.DICT)
            os.remove(In_root_tmp+file) # remove the image. 
            stopword_custom=[""," ","."]
            keywords= [w for w in data["text"] if w not in stopword_custom]
            df=pd.DataFrame({'keywords':keywords})
            df.to_csv(In_root_tmp+file[:-4]+".csv")


    

