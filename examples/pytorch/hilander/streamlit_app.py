from tkinter import image_names
from turtle import width
import streamlit as st
import pandas as pd
import numpy as np
#from deepface import DeepFace
import cv2
import numpy as np
import pandas as pd
#from imutils import paths
import os
import pickle
import sklearn
import pickle
from natsort import natsorted
import subprocess
import sys
import re
st.set_page_config(layout="wide")

#support function

def read_txt_file_path(path = '/Users/minhphu/Work/kltn/handcrawl2'):
    imgList = []
    data = []
    for root, dirs, files in os.walk(path):
        for f in files:
            #append the file name to the list
            full_path = os.path.join(root,f)
            if full_path.endswith('.txt'):
                imgList.append(full_path)
    #sorted(imgList, key=lambda x: x[0])
    imgList = natsorted(imgList)
    #print(imgList)
    return imgList
def convertResultToCSV_demo_handcrawl(path = None):
    #create DF and Result array
    #model, k, l, tau, pairwise_fscore,  bcubed_fscore, nmi, gt_cluster, pred_cluster, 
    #pairwise_precision, pairwise_recall, bcubed_precision, bcubed_recall, time, h_score, c_score, v_measure 
    df = pd.DataFrame(
        columns=['date_time','model_name','k','l','tau','pairwise_fscore','pairwise_precision','pairwise_recall',
    'bcubed_fscore','bcubed_precision','bcubed_recall','nmi','gt_cluster','pred_cluster'
    ,'cluster_time','eval_time','total_run_time','h_score','c_score','v_measure','test_name'])
    result = []
    with open(path, 'r') as f:
        for line in f.readlines():
            result.append(line.strip())
    def split_list(a_list):
        for i in range(0, len(a_list), 8):
            yield result[i:i + 8]
    result = list(split_list(result))
    #print(path)
    #preprocessing the file
    for res in result:
        baseDict = {} # create a dictionary
        date_time = res[0] # get the date_time
        baseDict['date_time'] = date_time # add the date_time to the dictionary
        #test_path = res[1].split(' ')[-1][3:]
        #baseDict['test_path'] = test_path
        k_level_info = res[1].split(' ')
        k_level_info = k_level_info[-1]# get the k_level_info
        #print(k_level_info)
        k_level_info = [int(s) for s in re.findall(r'\b\d+\b', k_level_info)]      
        k = int(k_level_info[0]) # get the k
        level = int(k_level_info[1]) # get the level
       # print(k, level)
        res.pop(0)
        res.pop(0)
        #res.pop(0) #remove date time, test path, k_level_info
        baseDict['k'] = k #add the k to the dictionary
        baseDict['l'] = level #add the level to the dictionary
        pairwiseDict = {} 
        bcubeDict = {}

        for i in range(0, len(res)):
            if i == (len(res)-3): #get cluster infor line
                temp = re.sub("[,:]","",res[i]).split(' ')
                #print(temp)
                for j in range(0, len(temp)):
                    temp[j] = re.sub("[,:#'}{]","",temp[j])
                #print(temp)
                baseDict['gt_cluster'] = int(temp[2])
                baseDict['pred_cluster'] = int(temp[5])
                baseDict['h_score'] = float(temp[7])
                baseDict['c_score'] = float(temp[9])
                baseDict['v_measure'] = float(temp[11])
            if i == (len(res)-2):
                baseDict['cluster_time'] = float(res[i])
            if i == (len(res)-1):
                baseDict['eval_time'] = float(res[i])
            temp = res[i].split(' ')
            name = temp.pop(0)
            name = re.sub('[:,]', '', name)
            for i in range(0, len(temp), 2): #get the pairwise, bcubed infor
                if name == 'pairwise':
                    pairwiseDict[re.sub("[,:]","",temp[i])] = float(re.sub("[,:]","",temp[i+1]))
                elif name == 'bcubed':
                    bcubeDict[re.sub("[,:]","",temp[i])] = float(re.sub("[,:]","",temp[i+1]))
                elif name == 'nmi':
                    baseDict['nmi'] = float(re.sub("[,:]}{","",temp[i]))   
        #print(pairwiseDict)
        baseDict['pairwise_precision'] = pairwiseDict['ave_pre']
        baseDict['pairwise_recall'] = pairwiseDict['ave_rec']
        baseDict['pairwise_fscore'] = pairwiseDict['fscore']
        baseDict['bcubed_precision'] = bcubeDict['ave_pre']
        baseDict['bcubed_recall'] = bcubeDict['ave_rec']
        baseDict['bcubed_fscore'] = bcubeDict['fscore']

        baseDict['total_run_time'] = baseDict['cluster_time'] + baseDict['eval_time']
        #get file name
        # if baseDict['pred_cluster'] == 251:
        #     baseDict['test_name'] = 'Hannah'
        # if baseDict['pred_cluster'] == 50289:
        #     baseDict['test_name'] = 'IMDB'
        # if baseDict['pred_cluster'] == 18084:
        #     baseDict['test_name'] = 'IMDB Same Distribution'
        baseDict['test_name'] = 'Handcrawl Demo'
        df_dict = pd.DataFrame.from_dict(baseDict, orient='index')
        #print(df_dict)
        #df = pd.concat(df_dict.T, ignore_index=True)
    return df_dict.T
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

#title
st.title('DEMO')
#header
#st.write('This is a demo of Streamlit')
#model = createModel('SFace')
#image_data = read_files('/Users/minhphu/Work/kltn/handcrawl2')
#createDataset(True)


if st.button('Create dataset'):
    #shell_env = sys.path
    #st.write(shell_env)
    my_env = os.environ.copy()
    list_env = my_env['PATH'].split(':')
    st.write(list_env)
    real_env = list_env[0].split('/')
    real_env[5] = 'tf_deepface'
    real_env = str.join('/', real_env)
    list_env[0] = real_env
    my_env['PATH'] = str.join(':', list_env)
    st.write(my_env['PATH'])
    #data_df = read_image_files('/Users/minhphu/Work/kltn/handcrawl2')
    subprocess.call('python createDataset.py', env=my_env, shell=True)
    st.write('Done')
    #st.dataframe(data_df)

#-----------------Clustering-----------------
if st.button('Clustering'):
    #run clustering, store result in handcrawl2 folder
    if os.system('python test_subg_demo_handcrawl.py --data_path /Users/minhphu/Work/kltn/handcrawl2/demo2.pkl --model_filename handcrawl_data/deepglint_random_sface_adam_2.pth --knn_k 10 --tau 0.5 --level 2 --threshold prob --hidden 512 --num_conv 1 --batch_size 16 --use_cluster_feat --early_stop'):
        st.success('This is a success message!', icon="✅")

#------------- Combine and show result -----------------
st.subheader('Combine and show result')

path = '/Users/minhphu/Work/kltn/handcrawl2' #path to folder contain images
data_df = read_image_files(path)
res_df = pd.read_csv('/Users/minhphu/Work/kltn/handcrawl2/demo_res.csv')
st.dataframe(res_df)
st.dataframe(data_df)
data_df['predicted'] = res_df['Predicted']
st.dataframe(data_df)

all_txt_path = read_txt_file_path()
#st.write(all_txt_path)
all_res_df = pd.DataFrame()
for path in all_txt_path:
    info = path.split('/')[-1].split('_')
    k = int(info[-4])
    level = int(info[-3])
    tau = float(info[-2])
    #print(k, level, tau)
    df = convertResultToCSV_demo_handcrawl(path)
    #print(df)
    #print(info[:-4])
    info_string = str.join('_', info[:-4])
    #print(info_string)
    df['model_name'] = info_string
    df['tau'] = tau
    #print(df.iloc[0])
    all_res_df = pd.concat([all_res_df, df], axis=0)

st.dataframe(all_res_df)

col_handcrawl1, col_handcrawl2,col_handcrawl3 = st.columns(3)
label_name = list(data_df['name'].unique())
label_name.append('All')
gt_label = list(data_df['labels'].unique())
gt_label.append('All')
pred_label = list(data_df['predicted'].unique())
pred_label.append('All')

#----------------- Select name -----------------
row1_spacer1, row1_1, row1_spacer2, row1_2, row1_spacer3  = st.columns((.2, 2.3, .4, 4.4, .2))
with row1_1:
    st.subheader('Select name')
    selected_label = st.selectbox('name',label_name, index = len(label_name)-1)
    if selected_label == 'All':
        st.warning('Please select a name')
    else:
        temp_df = data_df[data_df['name'] == selected_label]
        st.write('Selected name: ', selected_label)
        st.write('Number of images: ', len(data_df[data_df['name'] == selected_label]))
        pred_cluster = list(temp_df['predicted'].unique())
        pred_cluster.append('All')
        pred_num_cluster = len(temp_df['predicted'].unique())
        st.write('Number of predicted clusters: ', pred_num_cluster)
        choosen_cluster = st.selectbox('Choose pred cluster', pred_cluster, index = pred_num_cluster)
        st.write('Selected cluster: ', choosen_cluster)
        
with row1_2:
    if selected_label == 'All':
        st.warning('Please select a name')
    else:
        if choosen_cluster != 'All':
            temp_df = data_df[(data_df['name'] == selected_label) & (data_df['predicted'] == choosen_cluster)]
        else:
            temp_df = data_df[data_df['name'] == selected_label]
        st.dataframe(temp_df)
        for path in temp_df['path']:
            st.image(path, width=200)
        
#------Select predicted cluster------
row2_spacer1, row2_1, row2_spacer2, row2_2, row2_spacer3  = st.columns((.2, 2.3, .4, 4.4, .2))
with row2_1:
    st.subheader('Select predicted cluster')
    selected_pred_label = st.selectbox('predicted',pred_label, index = len(pred_label)-1)
    if selected_pred_label == 'All':
        st.warning('Please select a predicted cluster')
    else:
        temp_df = data_df[data_df['predicted'] == selected_pred_label]
        st.write('Selected predicted cluster: ', selected_pred_label)
        st.write('Number of images: ', len(data_df[data_df['predicted'] == selected_pred_label]))
        name = list(temp_df['name'].unique())
        name.append('All')
        name_num = len(temp_df['name'].unique())
        st.write('Number of names: ', name_num)
        for i in name:
            if i == 'All':
                continue
            st.markdown("- " + i + " (" + str(len(temp_df[temp_df['name'] == i])) + " images)")
         
        choosen_name = st.selectbox('Choose name', name, index = name_num)
        st.write('Selected name: ', choosen_name)

with row2_2:
    if selected_pred_label == 'All':
        st.warning('Please select a predicted cluster')
    else:
        if choosen_name != 'All':
            temp_df = data_df[(data_df['name'] == choosen_name) & (data_df['predicted'] == selected_pred_label)]
        else:
            temp_df = data_df[data_df['predicted'] == selected_pred_label]
        st.dataframe(temp_df)
        for path in temp_df['path']:
            st.image(path, width=200)



# col_handcrawl2,col_handcrawl3 = st.columns(3)
# with col_handcrawl1:
#     selected_label = st.selectbox('name',label_name, index = len(label_name)-1)
# with col_handcrawl2:
#     selected_pred_label = st.selectbox('predicted',pred_label, index = len(pred_label)-1)
# with col_handcrawl3:
#     selected_gt_label = st.selectbox('gt',gt_label, index = len(label_name)-1)




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