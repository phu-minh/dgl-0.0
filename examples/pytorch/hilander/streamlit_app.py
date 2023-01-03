from tkinter import image_names
from turtle import width
import streamlit as st
import pandas as pd
import numpy as np
#from deepface import DeepFace
import cv2
import numpy as np
#import pandas as pd
from imutils import paths
import os
import pickle
import sklearn
import pickle
from natsort import natsorted

st.set_page_config(layout="wide")

#support function
def read_files_in_folder(path):
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

def saveEncoding(data, path = '/Users/minhphu/Work/kltn/handcrawl',name = 'demo2.pkl'):
    with open(os.path.join(path,name),'wb') as f: 
        pickle.dump(data, f)
    return 

def createDataset(rerun = False):
    data_df = read_image_files('/Users/minhphu/Work/kltn/handcrawl2')
    #model = createModel('SFace')
    if os.path.exists('/Users/minhphu/Work/kltn/handcrawl/demo2.pkl'):
        st.write('Already have data! Located at /Users/minhphu/Work/kltn/handcrawl/demo2.pkl')
    if rerun:
        data = createEncodingPickl(data_df,model)
        saveEncoding(data)
        st.write('Done')
    return data_df

#title
st.title('DEMO')
#header
#st.write('This is a demo of Streamlit')
#model = createModel('SFace')
#image_data = read_files('/Users/minhphu/Work/kltn/handcrawl2')
#createDataset(True)



#-----------------Create dataset-----------------
if st.button('Clustering'):
    path = '/Users/minhphu/Work/kltn/handcrawl2' #path to folder contain images
    data_df = createDataset() #return dataframe contain path to images and labels, rerun = True to create dataset again
    #run clustering, store result in handcrawl2 folder
    if os.system('python test_subg_demo_handcrawl.py --data_path /Users/minhphu/Work/kltn/handcrawl/demo2.pkl --model_filename handcrawl_data/deepglint_random_sface_adam_2.pth --knn_k 10 --tau 0.5 --level 2 --threshold prob --hidden 512 --num_conv 1 --batch_size 16 --use_cluster_feat --early_stop'):
        st.success('This is a success message!', icon="✅")
    res_df = pd.read_csv('/Users/minhphu/Work/kltn/handcrawl2/demo_res.csv')
    st.dataframe(res_df)
    st.dataframe(data_df)
    # from read_result import convertResultToCSV_demo_handcrawl
    # st.write('Done')
    # all_df = pd.DataFrame()
    # all_path = read_files_in_folder('output2/')
    # for path in all_path:
    #     info = path.split('/')[-1].split('_')
    #     k = int(info[-4])
    #     level = int(info[-3])
    #     tau = float(info[-2])
    #     print(k, level, tau)
    #     df = convertResultToCSV_demo_handcrawl(path)
    #     #print(df)
    #     #print(info[:-4])
    #     info_string = str.join('_', info[:-4])
    #     print(info_string)
    #     df['model_name'] = info_string
    #     df['tau'] = tau
    #     #print(df.iloc[0])
    #     all_df = pd.concat([all_df, df], axis=0)





#-----------------Show all result-----------------
st.subheader('Show all result')
df = pd.read_csv('output2/demo.csv')
with st.expander('Click to show all result'):
    st.dataframe(df)

#-----------------Filter result-----------------
st.subheader('Filter result and show highest one')
#get all unique names in df 
model_names =list(df['model_name'].unique())
ks = list(df['k'].unique())
ls = list(df['l'].unique())
taus = list(df['tau'].unique())


model_names.append('All')
ks.append('All')
ls.append('All')
taus.append('All')

col1, col2,col3, col4 = st.columns(4)
st.write('Choose result to show')
with col1:
    selected_model = st.selectbox('Model',model_names, index = len(model_names)-1)
with col2:
    selected_knn = st.selectbox('Knn',ks, index = len(ks)-1)
with col3:
    selected_level = st.selectbox('Level',ls,index = len(ls)-1)
with col4:
    selected_tau = st.selectbox('Tau', taus, index = len(taus)-1)

if selected_model == 'All' and selected_knn == 'All' and selected_level == 'All' and selected_tau == 'All':
    st.write('Showing all result')
    with st.expander('Click to show all result'):
        st.dataframe(df)
else:
    #st.write(selected_model,selected_knn,selected_level,selected_tau)
    if selected_model == 'All':
        filtered_values = df[(df['k']==selected_knn) & (df['l']== selected_level) & (df['tau'] == selected_tau)]
        filtered_df = pd.DataFrame(filtered_values)
    elif selected_model != 'All':
        filtered_values = df[(df['model_name'] == selected_model)]
        filtered_df = pd.DataFrame(filtered_values)
    elif selected_knn == 'All':
        filtered_values = df[(df['model_name'] == selected_model) & (df['l']== selected_level) & (df['tau'] == selected_tau)]
        filtered_df = pd.DataFrame(filtered_values)
    elif selected_level == 'All':
        filtered_values = df[(df['k']==selected_knn) & (df['model_name'] == selected_model) & (df['tau'] == selected_tau)]
        filtered_df = pd.DataFrame(filtered_values)
    elif selected_tau == 'All':
        filtered_values = df[(df['k']==selected_knn) & (df['l']== selected_level) & (df['model_name'] == selected_model)]
        filtered_df = pd.DataFrame(filtered_values)
    else:
        filtered_values = df[(df['k']==selected_knn) & (df['l']== selected_level) & (df['model_name'] == selected_model) & (df['tau'] == selected_tau)]
        filtered_df = pd.DataFrame(filtered_values)
    if filtered_df.empty:
        st.warning('No result')
    else :
        with st.expander('Click to show filtered result'):
            st.dataframe(filtered_df)
        selected_metric = st.selectbox('Choose other metric',['nmi','pairwise','bcubed'],index=2)
        if st.button('Show '+str(selected_metric) +' highest metrics'):
            #with selected_metric:
            if selected_metric == 'bcubed':
                #get the highest bcubed_fscore
                max_bcubed_fscore = filtered_df['bcubed_fscore'].max()
                st.markdown('The **highest** bcubed_fscore is: ' +str(max_bcubed_fscore))
                max_bcubed_fscore_df = filtered_df[filtered_df['bcubed_fscore'] == max_bcubed_fscore]
                st.write('Model name: ' + str(max_bcubed_fscore_df['model_name'].values[0]))
                st.write('K: ' + str(max_bcubed_fscore_df['k'].values[0]), '    Level: ' + str(max_bcubed_fscore_df['l'].values[0]),'   Tau: ' + str(max_bcubed_fscore_df['tau'].values[0]))
                st.write('Test on: ' + str(max_bcubed_fscore_df['test_name'].values[0]))
                st.write('Number of ground truth clusters: ' + str(max_bcubed_fscore_df['gt_cluster'].values[0]))
                st.write('Number of predicted clusters: ' + str(max_bcubed_fscore_df['pred_cluster'].values[0]))
                st.write('Total running time: ' + str(max_bcubed_fscore_df['total_run_time'].values[0]) + ' seconds')
                with st.expander('Show detail'):
                    st.dataframe(max_bcubed_fscore_df)
            if selected_metric == 'nmi':
                max_nmi = filtered_df['nmi'].max()
                st.markdown('The **highest nmi** score is: ' +str(max_nmi))
                st.write('Detail of the highest nmi')
                max_nmi_df = filtered_df[filtered_df['nmi'] == max_nmi]
                st.write('Model name: ' + str(max_nmi_df['model_name'].values[0]))
                st.write('K: ' + str(max_nmi_df['k'].values[0]), '    Level: ' + str(max_nmi_df['l'].values[0]),'   Tau: ' + str(max_nmi_df['tau'].values[0]))
                st.write('Test on: ' + str(max_nmi_df['test_name'].values[0]))
                st.write('Number of ground truth clusters: ' + str(max_nmi_df['gt_cluster'].values[0]))
                st.write('Number of predicted clusters: ' + str(max_nmi_df['pred_cluster'].values[0]))
                st.write('Total running time: ' + str(max_nmi_df['total_run_time'].values[0]) + ' seconds')
                if st.button('Show detail'):
                    st.dataframe(max_nmi_df)
            elif selected_metric == 'pairwise':
                max_pairwise = filtered_df['pairwise_fscore'].max()
                st.write('The highest pairwise_fscore is: ' + str(max_pairwise))
                st.write('Detail of the highest pairwise_fscore')
                max_pairwise_df = filtered_df[filtered_df['pairwise_fscore'] == max_pairwise]
                st.write('Model name: ' + str(max_pairwise_df['model_name'].values[0]))
                st.write('K: ' + str(max_pairwise_df['k'].values[0]), '    Level: ' + str(max_pairwise_df['l'].values[0]),'   Tau: ' + str(max_pairwise_df['tau'].values[0]))
                st.write('Test on: ' + str(max_pairwise_df['test_name'].values[0]))
                st.write('Number of ground truth clusters: ' + str(max_pairwise_df['gt_cluster'].values[0]))
                st.write('Number of predicted clusters: ' + str(max_pairwise_df['pred_cluster'].values[0]))
                st.write('Total running time: ' + str(max_pairwise_df['total_run_time'].values[0]) + ' seconds')
                with st.expander('Show detail'):
                    st.dataframe(max_pairwise_df)

    

#st.image(image_paths[1])
#createDataset()

# df = pd.read_csv('result.csv')
# df_DBSCAN = pd.read_csv('resultDBSCAN.csv')
# st.button('Click here to see the result of the HILANDER algorithm')
# st.write('This is a detail table of the result of the HILANDER algorithm')
# #st.dataframe(df.drop('test_path', axis=1))
# st.write('This is a detail table of the result of the DBSCAN algorithm')
#st.dataframe(df_DBSCAN) 