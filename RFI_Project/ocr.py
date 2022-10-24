import cv2
import pytesseract
from pytesseract import Output
root="/media/mst/Backup/dataset/OCR"
img = cv2.imread(root+'/example2.png')

data = pytesseract.image_to_data(img, output_type=Output.DICT)
print(data["text"])
# extract words from a drawing set
# check the accuracy 
# match with the tagged RFIs

