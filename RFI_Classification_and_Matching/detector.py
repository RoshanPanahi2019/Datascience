from os import listdir
from matplotlib import pyplot as plt
from PIL import Image
from scipy.optimize import linear_sum_assignment
import cv2
import numpy as np
import pandas as pd

# Join discords
# Try this CNN-based template matching: https://arxiv.org/pdf/1903.07254.pdf
# Discussion: https://www.reddit.com/r/computervision/comments/lpt48t/symbol_spotting_using_image_processing/
def temp_matching_cv2(drawing_dir,tmp,tmp_name,threshold): # Source: https://docs.opencv.org/4.5.0/d4/dc6/tutorial_py_template_matching.html
    img_rgb = cv2.imread(drawing_dir)
    img_gray = cv2.cvtColor(img_rgb, cv2.COLOR_BGR2GRAY)
    template = cv2.imread(tmp,0)
    w, h = template.shape[::-1]
    res = cv2.matchTemplate(img_gray,template,cv2.TM_CCOEFF_NORMED)
    loc = np.where( res >= threshold)
    for pt in zip(*loc[::-1]):
        cv2.rectangle(img_rgb, pt, (pt[0] + w, pt[1] + h), (0,0,255), 4)
        cv2.putText(img_rgb,tmp_name,(pt[0], pt[1] -10 ),cv2.FONT_HERSHEY_SIMPLEX,.5,(255,0,0),1,lineType=cv2.LINE_AA)

    cv2.imwrite('/media/mst/Backup/dataset/OCR/images/res.jpeg',img_rgb)
    #img_rgb = cv2.resize(img_rgb, (960, 540)) 
    cv2.imshow("img",img_rgb)
    cv2.waitKey(0)
    plt.show()

def temp_matching_sift(drawing_dir,gt_toilet_dir):
    #RoI_original_img=cv2.imread("D:/OtherCodes/Measurment/Code/BIM/Code/GroundTruth/frame152.jpg")
    list_similarity=[]
    bb_left_corner_coordiante_list=[]
    list_features=[]
    list_cost=[]
    color=(0,0,0)
    stride=10
    
    drawing=cv2.imread(drawing_dir)
    gt=cv2.imread(gt_toilet_dir)
    m_d,n_d,_=drawing.shape
    m_g,n_g,_=gt.shape  
    slide_limit_x=n_d-n_g-1
    slide_limit_y=m_d-m_g-1
    
    sift = cv2.xfeatures2d.SIFT_create(600)  
    kps, des = sift.detectAndCompute(gt, None)
    f = des

    for j in range(0,slide_limit_y,stride):
        for i in range(0,slide_limit_x,stride):
            #print(i+j)
            drawing_clone = drawing.copy()
            box=[j,i,m_g,n_g]
            image = drawing_clone[box[0]: box[0] + box[2], box[1]: box[1] + box[3], :]
       
            sift = cv2.xfeatures2d.SIFT_create(600)  
            kps, des1 = sift.detectAndCompute(image, None)

            RoI_clone = cv2.rectangle(drawing_clone,(box[1],box[0]),(box[1] + box[3],box[0] + box[2]),color,3)
            #imS = cv2.resize(RoI_clone, (960, 540))     
            cv2.imshow("RoI_imgRoI_img",RoI_clone)
            cv2.waitKey(1)

            if des1 is None: continue
            f1 = des1
            #try:
            S = np.dot(f, np.transpose(f1))
            nrm_similarity=np.linalg.norm(S)

            C = 1 - np.abs(S)
            row_ind, col_ind_0 = linear_sum_assignment(C)                           
            C = C[row_ind, col_ind_0].sum()
            list_cost.append(C)
            
            bb_left_corner_coordiante_list.append(box)

            #except:

                #print("error in detect_corner.py => computing similarity and assignment")
         
    color=(0,255,0)
    match_index=np.argmin(list_cost, axis=0)
    print(list_cost[:5])
    box=bb_left_corner_coordiante_list[match_index]
    corner_coordinates=((box[1]+box[3]//2),(box[0]+box[2]//2))
    
    image = cv2.rectangle(drawing_clone,(box[1],box[0]),(box[1] + box[3],box[0] + box[2]),color,3)
    image = cv2.circle(image, corner_coordinates, 1, (0,0,255), 5) 
    #imS = cv2.resize(image, (960, 540))     
    cv2.imshow("RoI_imgRoI_img",image)
    cv2.waitKey(0)

    #corner_coordinates=(corner_coordinates[0]+box_corner_RoI[0],corner_coordinates[1]+box_corner_RoI[1])   
    return(corner_coordinates)

#===========================
if __name__=="__main__":
    root="/media/mst/Backup/dataset/OCR/images/"
    #df=pd.read_csv("/media/mst/Backup/dataset/OCR/images/annotation/"+"annotation.csv")
    gt_toilet_dir=root+"toilet_1"
    gt_toilet_2_dir=root+"toilet_2"
    gt_sink_dir=root+"sink_1"
    gt_sink_2_dir=root+"sink_2"
    gt_sink_3_dir=root+"sink_3"
    gt_sink_4_dir=root+"sink_4"
    gt_hand_dryer_dir=root+"hand_dryer_1"
    gt_hand_dryer_2_dir=root+"hand_dryer_2"
   
    gt_door_dir=root+"door_1"
    gt_door_2_dir=root+"door_2"
    gt_door_3_dir=root+"door_3"
    gt_door_4_dir=root+"door_4"
    gt_exit_1_dir=root+"exit_1"
    gt_exit_2_dir=root+"exit_2"
    gt_exit_3_dir=root+"exit_3"
    pdf_file="skanska-OST Current Drawings Thru ASI 11 02.28.22.pdf" # input the name of the drawing pdf you want to convert to images.
    drawing_dir="/media/mst/Backup/dataset/OCR/images/"+pdf_file[:-4]+"/"+"272c9888-f9aa-4c5c-89e2-6ce6ec82ff7f-005.jpg"
    drawing_dir= root+"drawing_region"
    #temp_matching_sift(drawing_dir,gt_toilet_dir)

    temp_matching_cv2(drawing_dir,gt_toilet_dir,'Toilet',threshold=.55)
    temp_matching_cv2(drawing_dir,gt_toilet_2_dir,'Toilet',threshold=.49)
    temp_matching_cv2(drawing_dir,gt_sink_dir,'Sink',threshold=.68)
    temp_matching_cv2(drawing_dir,gt_sink_2_dir,'Sink',threshold=.66)
    #temp_matching_cv2(drawing_dir,gt_sink_3_dir,'Sink',threshold=.68)
   # temp_matching_cv2(drawing_dir,gt_sink_4_dir,'Sink',threshold=.68)
    temp_matching_cv2(drawing_dir,gt_door_dir,'Door',threshold=.29)
   # temp_matching_cv2(drawing_dir,gt_door_2_dir,'Door',threshold=.2)
   # temp_matching_cv2(drawing_dir,gt_door_3_dir,'Door',threshold=.2)
   # temp_matching_cv2(drawing_dir,gt_door_4_dir,'Door',threshold=.2)
    temp_matching_cv2(drawing_dir,gt_hand_dryer_dir,'Hand_Dryer', threshold=.7)
    temp_matching_cv2(drawing_dir,gt_hand_dryer_2_dir,'Hand_Dryer',threshold=.61)
    #temp_matching_cv2(drawing_dir,gt_exit_1_dir,'Exit_Sign',threshold=.61)
    #temp_matching_cv2(drawing_dir,gt_exit_2_dir,'Exit_Sign',threshold=.68)
    #temp_matching_cv2(drawing_dir,gt_exit_3_dir,'Exit_Sign',threshold=.68)