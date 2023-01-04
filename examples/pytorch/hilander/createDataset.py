from tkinter import image_names
from turtle import width
import pandas as pd
import numpy as np
from deepface import DeepFace
import cv2
import numpy as np
import pandas as pd
from imutils import paths
import os
import pickle
from natsort import natsorted
def read_files_in_folder(path):
    imgList = []
    data = []
    for root, dirs, files in os.walk(path):
        for f in files:
            #append the file name to the list
            full_path = os.path.join(root,f)
            if full_path.endswith(('.DS_Store', '.csv','.pkl','.txt')):
                continue
            imgList.append(full_path)
    #sorted(imgList, key=lambda x: x[0])
    imgList = natsorted(imgList)
    data_df = pd.DataFrame(imgList)
    return data_df

def read_image_files(path):
    imgList = []
    data = []
    for root, dirs, files in os.walk(path):
        for f in files:
            #append the file name to the list
            full_path = os.path.join(root,f)
            if full_path.endswith(('.jpg', '.png', '.jpeg', '.JPG', '.PNG', '.JPEG')):
                imgList.append(full_path)
    imgList = natsorted(imgList)
    for path in imgList:
        temp = path.split('/')
        label = temp[-2]
        data.append([path,label])

    data_df = pd.DataFrame(data)
    data_df.columns = ['path', 'name']
    labels, uniques = pd.factorize(data_df['name'].tolist())
    data_df['labels'] = labels
    return data_df
def read_image_paths(path):
    imagePaths = list(paths.list_images(path))
    return imagePaths
def createModel(name='SFace'):
    model = DeepFace.build_model(name)
    return model

def createEncodingPickl(data_df,model):
    data_list = data_df.values.tolist()
    encodings = []
    labels = []
    data = np.empty(0)
    for i in range(len(data_list) ):
        try:
            encoding = DeepFace.represent(img_path = data_list[i][0] , model = model, detector_backend = 'mtcnn', align = True, normalization = 'base')
            #encoding = np.linalg.norm(encoding)
            encodings.append(encoding)
            print(data_list[i][2])
            labels.append(data_list[i][2]) 
        except:
            print(data_list[i][0])
            continue
    encodings = np.array(encodings)
    labels = np.array(labels) 
    data = [encodings, labels]
    print(data[0].shape)
    print(data[1].shape)
    return data

def saveEncoding(data, path = '/Users/minhphu/Work/kltn/handcrawl2',name = 'demo2.pkl'):
    with open(os.path.join(path,name),'wb') as f: 
        pickle.dump(data, f)
    return 

def createDataset(rerun = False):
    data_df = read_image_files('/Users/minhphu/Work/kltn/handcrawl2')
    model = createModel('SFace')
    if os.path.exists('/Users/minhphu/Work/kltn/handcrawl2/demo2.pkl'):
        print('Already have data! Located at /Users/minhphu/Work/kltn/handcrawl2/demo2.pkl')
    if rerun:
        data = createEncodingPickl(data_df,model)
        saveEncoding(data)
        #st.write('Done')
    return data_df

if __name__ == '__main__':
    createDataset(rerun = True)