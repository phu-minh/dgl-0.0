import faulthandler
import hdbscan

from utils.knn import knn
faulthandler.enable()
import numpy as np
import pickle
from sklearn.cluster import DBSCAN
from utils import (
    build_knns,
    evaluation ,
    fast_knns2spmat,
    knns2ordered_nbrs,
    l2norm,
    row_normalize,
    sparse_mx_to_indices_values,
)
k = 10
# Load data

#real_train_data_path = 'data/subcenter_arcface_deepglint_train_1_in_10_recreated.pkl'
#real_test_data_path = 'data/subcenter_arcface_deepglint_imdb_features_sampled_as_deepglint_1_in_10.pkl'

real_train_data_path = 'data/deepglint_100classes_random30to50.pkl'
real_test_data_path = 'data/imdb_100classes_random30to50.pkl'
#data_path = 'handcrawl_data/train_encodings.pickle'
with open(real_train_data_path, "rb") as f:
    features, labels = pickle.load(f)
with open(real_test_data_path, "rb") as f:
    test_features, test_labels = pickle.load(f)
def knn_graph(features, k): 
    features = l2norm(features.astype("float32"))
    knns = build_knns(features, k, "faiss")
    dists, nbrs = knns2ordered_nbrs(knns)
    adj = fast_knns2spmat(knns, k) #create sparse matrix (ma trận kề)
    #print(adj)
    adj, adj_row_sum = row_normalize(adj) #calculate row sum (tính tổng các hàng)
    #print(adj, adj_row_sum)
    indices, values, shape = sparse_mx_to_indices_values(adj) #convert sparse matrix to indices, values, shape
    return features, adj, adj_row_sum

train_features, train_adj, _ = knn_graph(features, k)

cluster = hdbscan.HDBSCAN(metric='precomputed',alpha=0.2)

cluster.fit(train_adj,labels)

test_features, test_adj, _ = knn_graph(test_features, k)
test_labels = cluster.fit_predict(test_adj)

#print(test_adj)
evaluation(test_labels, test_labels, "pairwise,bcubed,nmi",'outputHDBSCAN.txt')