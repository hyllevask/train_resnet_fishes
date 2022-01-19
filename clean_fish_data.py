import cv2
import pandas as pd
from pandas.core.frame import DataFrame


#dataframe = pd.read_csv("./annotations/fungi_bb_annotation.csv")
#dataframe = pd.read_csv("./annotations/wild_farmed_bb_annotation.csv")
#dataframe.head()

dataframe = pd.read_csv("./annotations/combined_annotation.csv")


name_list = []
label_list = []
label2_list = []

N = len(dataframe.index)
margin = -10
index = 0
for ii in range(N):
    
    ind,name,bboxes = dataframe.iloc[ii]
    #print(name)
    bboxes = eval(bboxes)
    for bbox in bboxes:
        print(index)
        save_name = "./combined_images/image_"+str(index)+".jpg"
        hight = round(bbox['height'])
        label = bbox['label']
        label2 = bbox['label2']
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
            label_list.append(label)
            label2_list.append(label2)
            name_list.append("image_"+str(index)+".jpg")
        except:
            print("An error was raised")
        # Destroying present windows on screen
        cv2.destroyAllWindows() 
        index +=1

df2 = pd.DataFrame({'Filename':name_list,'label':label_list,'label2':label2_list})
df2.to_csv('./combined_images/annotations.csv')
    #im = read_and_crop(name,bbox,margin)
    