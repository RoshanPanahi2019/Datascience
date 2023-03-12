# This program takes pdf drawings and outputs the texts for each sheet. 
#TOD0: 
#   pyInstaller fr OCR
#   https://realpython.com/python-web-applications/
#   test match_RFI_dwg 
#   pyInstaller

from os import listdir, mkdir
import pytesseract
from pytesseract import Output
from pdf2image import convert_from_path
from PIL import Image
import os.path
import pandas as pd
import tkinter as tk
from tkinter import filedialog

# Ask user to open the drawing set 
def open(mymode):
    root = tk.Tk()
    root.withdraw()
    if mymode=='drawing': path = filedialog.askopenfilename(filetypes=[('PDF','.pdf')],title='Select Drawing')
    if mymode=='output': path = filedialog.askdirectory(title='Select Output Directory')
    if mymode=='tesseract': path = filedialog.askopenfilename(filetypes=[('EXE','.exe')],title='Select Tesseract Directory')
    return(path)

# Shorthand to create directories
def create_dir(dir):
    try: 
         os.mkdir(dir)
    except:
         print ("Path exists!")
         return 

# Convert the pdf file of the drawing to images and store them. 
def pdf_to_img(pdf_dir,img_dir):
    create_dir(img_dir)
    convert_from_path(pdf_path=pdf_dir,fmt='jpg',output_folder=img_dir)
    return 
    
# Extract the texts from each image and store them in a seperate file for each image.
def img_to_txt(img_dir,txt_dir):
    try :
        os.mkdir(txt_dir)
    except:
        print("The corpus already exists!")
        return
        
    for file in sorted(listdir(img_dir)):
        img = Image.open(img_dir+file)
        data = pytesseract.image_to_data(img, output_type=Output.DICT)
        stopword_custom=[""," ","."]
        keywords= [w for w in data["text"] if w not in stopword_custom]
        df=pd.DataFrame({'keywords':keywords})
        df.to_csv(txt_dir+file[:-4]+".csv")

#=============================
if __name__=="__main__":
    #drawing_file=open('drawing')
    #out_root=open('output')

    ## If you are using the wbesite:
    drawing_file="/media/ms/D/myGithub/Datascience/RFI_Classification_and_Matching/Website/static/files/ClearCreek_Drawings.pdf"
    out_root="/media/ms/D/myGithub/Datascience/RFI_Classification_and_Matching/Website/static/out/"
    head_tail=os.path.split(drawing_file)
    head=head_tail[1]
    create_dir(out_root+"/images/")
    img_dir=out_root+"/images/"+head[:-4]+"/"
    create_dir(out_root+"/corpus/")
    corpus_dir=out_root+"/corpus/"+head[:-4]+"/"
    
    #pytesseract.pytesseract.tesseract_cmd =r"C:\Program Files\Tesseract-OCR\tesseract.exe"
    #tesseract_path =open('tesseract')
    #pytesseract.pytesseract.tesseract_cmd=tesseract_path
    pdf_to_img(drawing_file,img_dir)
    img_to_txt(img_dir,corpus_dir)
