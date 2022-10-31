# This program takes pdf drawings and outputs the texts for each sheet. 

from os import listdir, mkdir
import cv2
import pytesseract
import nltk
from pytesseract import Output
from pdf2image import convert_from_path
from PIL import Image
import os.path
import pandas as pd

# Convert the pdf file of the drawing to images and store them. 
def pdf_to_img(pdf_dir,img_dir):
    try: 
         os.mkdir(img_dir)
    except:
         print ("The images have been extracted previously!")
         return 
    images = convert_from_path(pdf_path=pdf_dir)
    for i in range(len(images)):
        images[i].save(img_dir+str(i) +'.jpg')
    return 

# Extract the texts from each image and store them in a seperate file for each image.
def img_to_txt(img_dir,txt_dir):
    try :
        os.mkdir(txt_dir)
    except:
        print("The corpus already exists!")
        return
        
    for file in listdir(img_dir):
        img = Image.open(img_dir+file)
        data = pytesseract.image_to_data(img, output_type=Output.DICT)
        stopword_custom=[""," ","."]
        keywords= [w for w in data["text"] if w not in stopword_custom]
        df=pd.DataFrame({'keywords':keywords})
        df.to_csv(txt_dir+file[:-4]+".csv")

#=============================
if __name__=="__main__":
    root="/media/mst/Backup/dataset/OCR/"
    pdf_file="Crooked_Cogswell_Plans.pdf" # input the name of the drawing pdf you want to convert to images.
    for pdf_file in listdir(root+"drawings/"):
        pdf_dir=root+"drawings/"+pdf_file
        img_dir=root+"images/"+pdf_file[:-4]+"/"
        txt_dir=root+"corpus/"+pdf_file[:-4]+"/"
        pdf_to_img(pdf_dir,img_dir)
        img_to_txt(img_dir,txt_dir)



    
