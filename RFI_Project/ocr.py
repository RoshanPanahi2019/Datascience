import cv2
import pytesseract
from pytesseract import Output
import pandas as pd
root="/media/ms/D/myGithub_Classified/Skanska/Data"
dst="/media/ms/D/myGithub_Classified/Skanska/Data"
img = cv2.imread(root+'/floorplan1.png')
df=pd.DataFrame
df = pytesseract.image_to_data(img, output_type=Output.DICT)
print(df["text"])

# extract words from a drawing set using more accuracte method, use other library, ML based?
# check the accuracy 
# match with the tagged RFIs

