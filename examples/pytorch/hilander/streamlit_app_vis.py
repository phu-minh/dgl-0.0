from tkinter import image_names
from turtle import width
import streamlit as st
import pandas as pd
import numpy as np
from deepface import DeepFace
import cv2
import numpy as np
#import pandas as pd
from imutils import paths
import os
import pickle
import sklearn
import pickle
from natsort import natsorted

def read_files(path):
    imgList = []
    data = []
    for root, dirs, files in os.walk(path):
        for f in files:
            #append the file name to the list
            full_path = os.path.join(root,f)
            if full_path.endswith('.DS_Store'):
                continue
            imgList.append(full_path)
    #sorted(imgList, key=lambda x: x[0])
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
    #len(imagePaths)
    return imagePaths
def createModel(name='SFace'):
    model = DeepFace.build_model(name)
    return model

def createEncodingPickl(data_df,model):
    data_list = data_df.values.tolist()
   # st.write(data_list[0][2])
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

def saveEncoding(data, path = '/Users/minhphu/Work/kltn/handcrawl',name = 'demo2.pkl'):
    with open(os.path.join(path,name),'wb') as f: 
        pickle.dump(data, f)
    return 

def createDataset(rerun = False):
    data_df = read_files('/Users/minhphu/Work/kltn/handcrawl2')
    model = createModel('SFace')
    if os.path.exists('/Users/minhphu/Work/kltn/handcrawl/demo2.pkl'):
        st.write('Already have data!')
    if rerun:
        data = createEncodingPickl(data_df,model)
        saveEncoding(data)
        st.write('Done')
    return

#title
st.title('DEMO')
#header
#st.write('This is a demo of Streamlit')
# model = createModel('SFace')
# image_data = read_files('/Users/minhphu/Work/kltn/handcrawl2')
# st.write(image_data)
# image = cv2.imread(image_data['path'][0])
# # df = DeepFace.find(img_path = "/Users/minhphu/Work/kltn/handcrawl2/mbappe/01.jpg",
# #       db_path = "/Users/minhphu/Work/kltn/handcrawl2", 
# #       model_name = 'SFace',
# #       enforce_detection=False
# # )
# face = DeepFace.detectFace(img_path = "/Users/minhphu/Work/kltn/handcrawl2/mbappe/01.jpg", 
#         target_size = (224, 224), 
#         detector_backend = 'mtcnn',
# )
# st.image(face)
if st.button('Get Run result'):
    df = pd.read_csv('output2/demo.csv')
    st.dataframe(df)

    #get all unique names in df 
    model_names = df['model_name'].unique()
    ks = df['k'].unique()
    ls = df['l'].unique()
    taus = df['tau'].unique()

    col1, col2,col3, col4 = st.columns(4)
    st.write('Choose result to show')
    with col1:
        selected_model = st.selectbox('Model',model_names)
    with col2:
        selected_knn = st.selectbox('Knn',ks)
    with col3:
        selected_level = st.selectbox('Level',ls)
    with col4:
        selected_tau = st.selectbox('Tau',taus)

    if st.button('Show'):
        filtered_values = df[(df['k']==selected_knn) & (df['l']== selected_level) & (df['model_name'] == selected_model) & (df['tau'] == selected_tau)]
        filtered_df = pd.DataFrame(filtered_values)
        st.dataframe(filtered_df)
        #
