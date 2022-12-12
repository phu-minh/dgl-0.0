from math import radians
import pickle
import numpy
from utils.knn import knn
import faulthandler

import numpy as np
import pickle
from sklearn.cluster import DBSCAN
from utils import (
    adjacency,
    build_knns,
    evaluation ,
    fast_knns2spmat,
    knns2ordered_nbrs,
    l2norm,
    row_normalize,
    sparse_mx_to_indices_values,
)
faulthandler.enable()

#data/subcenter_arcface_deepglint_train_1_in_10_recreated.pkl


k=10

def knn_graph(features, k):
    features = l2norm(features.astype("float32"))
    knns = build_knns(features, k, "faiss_gpu")
    dists, nbrs = knns2ordered_nbrs(knns)
    adj = fast_knns2spmat(knns, k) #create sparse matrix (ma trận kề)
    #print(adj)
    adj, adj_row_sum = row_normalize(adj) #calculate row sum (tính tổng các hàng)
    #print(adj, adj_row_sum)
    indices, values, shape = sparse_mx_to_indices_values(adj) #convert sparse matrix to indices, values, shape
    list_adj = adj.tolil().astype(np.float32)
    adjacency_list = []
    for row in list_adj:
        nonzero = np.nonzero(row.toarray())
        its_self = [nonzero[0][0]]
        neighbor = nonzero[1]
        its_self.extend(neighbor)
        adjacency_list.append(its_self)
    return features, adj, adj_row_sum, adjacency_list


with open('/content/drive/MyDrive/KLTN_DATA/dgl/examples/pytorch/hilander/data/subcenter_arcface_deepglint_train_1_in_10_recreated.pkl', 'rb') as f:
    features, labels = pickle.load(f)
features, adj, _,adjacency_list = knn_graph(features, k)
#save to npy file
print(labels.shape)
print(features.shape)
print(adjacency_list[0])
numpy.save('npy/train_deepglint_features.npy', features)
numpy.save('npy/train_deepglint_labels.npy', labels)
numpy.save('npy/train_deepglint_knn.npy', adjacency_list)

del labels
del features
del adjacency_list




with open('/content/drive/MyDrive/KLTN_DATA/dgl/examples/pytorch/hilander/data/subcenter_arcface_deepglint_imdb_features_sampled_as_deepglint_1_in_10.pkl', 'rb') as f:
    test_features, test_labels = pickle.load(f)

test_features, test_adj, _,test_adjacency_list = knn_graph(test_features, k)
print(test_labels.shape)
print(test_features.shape)
print(test_adjacency_list[0])
numpy.save('npy/test_imdb_same_dist_features.npy', test_features)
numpy.save('npy/test_imdb_same_dist_labels.npy', test_labels)
numpy.save('npy/test_imdb_same_dist_knn.npy', test_adjacency_list)

del test_adj
del test_features
del test_adjacency_list




with open('/content/drive/MyDrive/KLTN_DATA/dgl/examples/pytorch/hilander/data/subcenter_arcface_deepglint_imdb_features.pkl', 'rb') as f:
    test_features1, test_labels1 = pickle.load(f)      
test_features1, test_adj1, _, test_adjacency_list1 = knn_graph(test_features1, k)
print(test_labels1.shape)
print(test_features1.shape)
print(test_adjacency_list1[0])
numpy.save('npy/test_imdb_features.npy', test_features1)
numpy.save('npy/test_imdb_labels.npy', test_labels1)
numpy.save('npy/test_imdb_knn.npy', test_adjacency_list1)
del test_adj1
del test_features1
del test_adjacency_list1

# with open('/content/drive/MyDrive/KLTN_DATA/dgl/examples/pytorch/hilander/data/subcenter_arcface_deepglint_hannah.pkl', 'rb') as f:
#     test_features2, test_labels2 = pickle.load(f)      
# test_features2, test_adj2, _, test_adjacency_list2 = knn_graph(test_features2, k)
# print(test_labels2.shape)
# print(test_features2.shape)
# print(test_adjacency_list2[0])
# numpy.save('npy/test_hannah_features.npy', test_features2)
# numpy.save('npy/test_hannah_labels.npy', test_labels2)
# numpy.save('npy/test_hannah_knn.npy', test_adjacency_list2)

# del test_adj2
# del test_features2
# del test_adjacency_list2

    