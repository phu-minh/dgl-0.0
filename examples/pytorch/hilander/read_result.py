from cgi import test
import json
import re
import pandas as pd

def convertResultToCSV(path='output.txt'):
    #create DF and Result array
    df = pd.DataFrame(columns=['date_time','test_path', 'k_knn','level','pairwise','bcubed','nmi','gt_cluster','pred_cluster','h_score','c-score','v_meansure'])
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
    df.to_csv('result.csv', index=False)
    return df

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
        nmiDict = {} 
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

convertResultToCSV()
#convertResultToCSV_DBSCAN()
