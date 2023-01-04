from cgi import test
import json
import re
import pandas as pd
import os
from natsort import natsorted
def read_files_in_folder(path):
    imgList = []
    data = []
    for root, dirs, files in os.walk(path):
        for f in files:
            #append the file name to the list
            full_path = os.path.join(root,f)
            if full_path.endswith('.DS_Store'):
                continue
            if full_path.endswith('.csv'):
                continue
            imgList.append(full_path)
    #sorted(imgList, key=lambda x: x[0])
    imgList = natsorted(imgList)
    #print(imgList)
    return imgList

def convertResultToCSV(path='output.txt'):
    #create DF and Result array
    df = pd.DataFrame(columns=['date_time','test_path', 'k_knn','level','pairwise','bcubed','nmi','gt_cluster','pred_cluster','h_score','c-score','v_meansure'])
    result = []
    with open(path, 'r') as f:
        for line in f.readlines():
            result.append(line.strip())
    def split_list(a_list):
        for i in range(0, len(a_list), 8):
            yield result[i:i + 8]
    result = list(split_list(result))
    print(result)
    #preprocessing the file
    for res in result:
        baseDict = {} # create a dictionary
        date_time = res[0] # get the date_time
        baseDict['date_time'] = date_time # add the date_time to the dictionary
        test_path = res[1].split(' ')[-1][3:]
        baseDict['test_path'] = test_path
        k_level_info = (res[2].split(' '))[-1] # get the k_level_info
        k_level_info = [int(s) for s in re.findall(r'\b\d+\b', k_level_info)]      
        k = int(k_level_info[0]) # get the k
        level = int(k_level_info[1]) # get the level
        print(k, level)
        res.pop(0)
        res.pop(0)
        res.pop(0) #remove date time, test path, k_level_info
        baseDict['k_knn'] = k #add the k to the dictionary
        baseDict['level'] = level #add the level to the dictionary
        pairwiseDict = {} 
        bcubeDict = {}
        for i in range(0, len(res)):
            if i == (len(res)-1):
                temp = re.sub("[,:]","",res[i]).split(' ')
                for j in range(0, len(temp)):
                    temp[j] = re.sub("[,:#'}{]","",temp[j])
                #print(temp)
                baseDict['pred_cluster'] = int(temp[2])
                baseDict['gt_cluster'] = int(temp[5])
                baseDict['h_score'] = float(temp[7])
                baseDict['c-score'] = float(temp[9])
                baseDict['v_meansure'] = float(temp[11])
            temp = res[i].split(' ')
            name = temp.pop(0)
            name = re.sub('[:,]', '', name)
            for i in range(0, len(temp), 2):
                if name == 'pairwise':
                    pairwiseDict[temp[i]] = float(re.sub("[,:]","",temp[i+1]))
                elif name == 'bcubed':
                    bcubeDict[temp[i]] = float(re.sub("[,:]","",temp[i+1]))
                elif name == 'nmi':
                    baseDict['nmi'] = float(re.sub("[,:]}{","",temp[i]))                
        baseDict['pairwise'] = pairwiseDict
        baseDict['bcubed'] = bcubeDict
        df_dict = pd.DataFrame.from_dict(baseDict, orient='index')
        #print(df_dict)
        df = df.append(df_dict.T, ignore_index=True)
    # #df.to_csv('result.csv', index=False)
    # return df
    return 

def convertResultToCSV_DBSCAN(path='outputDBSCAN.txt'):
    df = pd.DataFrame(columns=['date_time', 'k_knn','train_path','test_path','pairwise','bcubed','nmi','gt_cluster','pred_cluster','h_score','c-score','v_meansure'])
    result = []
    with open(path, 'r') as f:
        for line in f.readlines():
            result.append(line.strip())
    def split_list(a_list):
        for i in range(0, len(a_list), 7):
            yield result[i:i + 7]
    result = list(split_list(result))
    #preprocessing the file
    for res in result:
        baseDict = {} # create a dictionary
        date_time = res[0] # get the date_time
        baseDict['date_time'] = date_time # add the date_time to the dictionary
        res.pop(0) #remove date time
        # k_level_info = (res[1].split(' '))[-1] # get the k_level_info
        # k = int(k_level_info[2]) # get the k
        # level = int(k_level_info[-1]) # get the level
        train_path = res[0].split(' ')[-1]
        res.pop(0)
        test_path = res[0].split(' ')[-1]
        res.pop(0) #remove paths
        baseDict['train_path'] = train_path
        baseDict['test_path'] = test_path
        baseDict['k_knn'] = 10 #add the k to the dictionary
        pairwiseDict = {} 
        bcubeDict = {}
        #nmiDict = {} 
        for i in range(0, len(res)):
            if i == (len(res)-1):
                temp = re.sub("[,:]","",res[i]).split(' ')
                for j in range(0, len(temp)):
                    temp[j] = re.sub("[,:#'}{]","",temp[j])
                baseDict['pred_cluster'] = int(temp[2])
                baseDict['gt_cluster'] = int(temp[5])
                baseDict['h_score'] = float(temp[7])
                baseDict['c-score'] = float(temp[9])
                baseDict['v_meansure'] = float(temp[11])
            temp = res[i].split(' ')
            name = temp.pop(0)
            name = re.sub('[:,]', '', name)
            for i in range(0, len(temp), 2):
                if name == 'pairwise':
                    pairwiseDict[temp[i]] = float(re.sub("[,:]","",temp[i+1]))
                elif name == 'bcubed':
                    bcubeDict[temp[i]] = float(re.sub("[,:]","",temp[i+1]))
                elif name == 'nmi':
                    baseDict['nmi'] = float(re.sub("[,:]}{","",temp[i]))
        baseDict['pairwise'] = pairwiseDict
        baseDict['bcubed'] = bcubeDict
        df_dict = pd.DataFrame.from_dict(baseDict, orient='index')
        #print(df_dict)
        df = df.append(df_dict.T, ignore_index=True)
    df.to_csv('resultDBSCAN.csv', index=False)
    return df

def convertResultToCSV_demo(path):
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
    
    #preprocessing the file
    for res in result:
        baseDict = {} # create a dictionary
        date_time = res[0] # get the date_time
        baseDict['date_time'] = date_time # add the date_time to the dictionary
        #test_path = res[1].split(' ')[-1][3:]
        #baseDict['test_path'] = test_path
        k_level_info = (res[1].split(' '))[-1] # get the k_level_info
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
                baseDict['pred_cluster'] = int(temp[2])
                baseDict['gt_cluster'] = int(temp[5])
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
        if baseDict['pred_cluster'] == 251:
            baseDict['test_name'] = 'Hannah'
        if baseDict['pred_cluster'] == 50289:
            baseDict['test_name'] = 'IMDB'
        if baseDict['pred_cluster'] == 18084:
            baseDict['test_name'] = 'IMDB Same Distribution'
        df_dict = pd.DataFrame.from_dict(baseDict, orient='index')
        #print(df_dict)
        df = df.append(df_dict.T, ignore_index=True)
    return df


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
        print(k_level_info)
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
        if baseDict['pred_cluster'] == 251:
            baseDict['test_name'] = 'Hannah'
        if baseDict['pred_cluster'] == 50289:
            baseDict['test_name'] = 'IMDB'
        if baseDict['pred_cluster'] == 18084:
            baseDict['test_name'] = 'IMDB Same Distribution'
        df_dict = pd.DataFrame.from_dict(baseDict, orient='index')
        #print(df_dict)
        df = df.append(df_dict.T, ignore_index=True)
    return df
    
#convertResultToCSV()xw
#convertResultToCSV_DBSCAN()
all_df = pd.DataFrame()
all_path = read_files_in_folder('output2/')
for path in all_path:
    info = path.split('/')[-1].split('_')
    k = int(info[-4])
    level = int(info[-3])
    tau = float(info[-2])
    print(k, level, tau)
    df = convertResultToCSV_demo_handcrawl(path)
    #print(df)
    #print(info[:-4])
    info_string = str.join('_', info[:-4])
    print(info_string)
    df['model_name'] = info_string
    df['tau'] = tau
    #print(df.iloc[0])
    all_df = pd.concat([all_df, df], axis=0)

#print(all_df)
all_df.to_csv('output2/demo.csv', index=False)
#model, k, l, tau, pairwise_fscore,  bcubed_fscore, nmi, gt_cluster, pred_cluster, pairwise_precision, pairwise_recall, bcubed_precision, bcubed_recall, time, h_score, c_score, v_measure 