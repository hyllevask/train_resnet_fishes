import cv2
import pandas as pd
import numpy as np
from pandas.core.frame import DataFrame

def compare_bboxes(bboxes1,bboxes2,th):
    # determine the (x, y)-coordinates of the intersection rectangle
    # Expects the format 
    
    out_list = []
    for ii,box1 in enumerate(bboxes1):
        bA = [box1['left'],box1['top'],box1['left']+box1['width'],box1['top']+box1['height']]
        score = []
        for jj,box2 in enumerate(bboxes2):
            bB = [box2['left'],box2['top'],box2['left']+box2['width'],box2['top']+box2['height']]

            score.append(bb_intersection_over_union(bA,bB))
        #Find index of max score
        max_index = np.argmax(score)
        if score[max_index] > th:
            box1['label2'] = bboxes2[max_index]['label']
            out_list.append(box1) 


    return out_list

def bb_intersection_over_union(boxA, boxB):
    # determine the (x, y)-coordinates of the intersection rectangle
    #Expects coordinates in the form [x0,y0,x1,y1]
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    # compute the area of intersection rectangle
    interArea = (xB - xA) * (yB - yA)

    # compute the area of both the prediction and ground-truth
    # rectangles
    boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])

    # compute the intersection over union by taking the intersection
    # area and dividing it by the sum of prediction + ground-truth
    # areas - the interesection area
    iou = interArea / float(boxAArea + boxBArea - interArea)

    # return the intersection over union value
    return iou






dataframe1 = pd.read_csv("./annotations/wild_farmed_bb_annotation.csv")
dataframe2 = pd.read_csv("./annotations/fungi_bb_annotation.csv")

col_names = ["Image","annotation"]
matching_dataframe = DataFrame(columns=col_names)

name_list = []
lable_list = []

N = len(dataframe1.index)
index = 0
N_running = 0
for ii in range(N):
    
    name,bboxes,flag = dataframe1.iloc[ii]
    test = dataframe2[dataframe2["Image.name"] == name]
    #print(test)
    if not test.empty:
        N_running += 1
        print(N_running)
        name2,bboxes2,flag2 = test.iloc[0]
        out = compare_bboxes(eval(bboxes),eval(bboxes2),th=0.7)
        #Returns a list of dictionaries
        dd = {"Image":name,"annotation":str(out)}
        matching_dataframe = matching_dataframe.append(dd,ignore_index=True)
        #print(out)

    
matching_dataframe.to_csv("combined_annotation.csv")
    # Search in dataframe2 for the matching 

'''

    #print(name)
    bboxes = eval(bboxes)
    for bbox in bboxes:
        print(index)
        save_name = "./fin_images/image_"+str(index)+".jpg"
        hight = round(bbox['height'])
        lable = bbox['label']
        left = round(bbox['left'])
        top = round(bbox['top'])
        width = round(bbox['width'])
        im = cv2.imread('./' + name)

        im2 = im[top-margin:top+hight+margin,left-margin:left+width+margin]
        #cv2.imshow('image',im[top-margin:top+hight+margin,left-margin:left+width+margin])
        #cv2.waitKey(1000)        
        if im2.__len__() == 0:
            continue
        try:
            cv2.imwrite(save_name,im2)
            lable_list.append(lable)
            name_list.append("image_"+str(index)+".jpg")
        except:
            print("An error was raised")
        # Destroying present windows on screen
        cv2.destroyAllWindows() 
        index +=1

df2 = pd.DataFrame({'Filename':name_list,'lable':lable_list})
df2.to_csv('./fin_images/annotations.csv')
    #im = read_and_crop(name,bbox,margin)
'''