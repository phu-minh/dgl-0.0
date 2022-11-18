# import dgl
# import matplotlib.pyplot as plt
# import networkx as nx
# import torch

# g = dgl.graph(([0, 1, 2, 3], [1, 2, 3, 4]))
# g.ndata['h'] = torch.ones(5, 2)
# g.apply_edges(lambda edges: {'x' : edges.src['h'] + edges.dst['h']})
# g.edata['x']

# nx_g = dgl.to_networkx(g, node_attrs=['h'], edge_attrs=['x'])
# nx_g.nodes(data=True)
# options = {
#     'node_color': 'black',
#     'node_size': 20,
#     'width': 1,
# }
# plt.figure(figsize=[15,7])
# nx.draw(nx_g, **options)

# import math
# import multiprocessing as mp
# import os
# import pickle
# import numpy as np
# from tqdm import tqdm
# from utils import Timer

# from utils.faiss_search import faiss_search_knn

# def build_knns(feats, k, knn_method, dump=True):
#     with Timer("build index"):
#         # build index for knn search
#         # and search for k-nearest neighbors
#         if knn_method == "faiss":
#             index = knn_faiss(feats, k, omp_num_threads=None) 
#         # elif knn_method == "faiss_gpu":
#         #     index = knn_faiss_gpu(feats, k)
#         else:
#             raise KeyError(
#                 "Only support faiss and faiss_gpu currently ({}).".format(
#                     knn_method
#                 )
#             )
#         knns = index.get_knns()
#     return knns


# class knn:
#     def __init__(self, feats, k, index_path="", verbose=True):
#         pass

#     def filter_by_th(self, i):
#         th_nbrs = []
#         th_dists = []
#         nbrs, dists = self.knns[i]
#         for n, dist in zip(nbrs, dists):
#             if 1 - dist < self.th:
#                 continue
#             th_nbrs.append(n)
#             th_dists.append(dist)
#         th_nbrs = np.array(th_nbrs)
#         th_dists = np.array(th_dists)
#         return (th_nbrs, th_dists)

#     def get_knns(self, th=None):
#         if th is None or th <= 0.0:
#             return self.knns
#         # TODO: optimize the filtering process by numpy
#         # nproc = mp.cpu_count()
#         nproc = 1
#         with Timer(
#             "filter edges by th {} (CPU={})".format(th, nproc), self.verbose
#         ):
#             self.th = th
#             self.th_knns = []
#             tot = len(self.knns)
#             if nproc > 1:
#                 pool = mp.Pool(nproc)
#                 th_knns = list(
#                     tqdm(pool.imap(self.filter_by_th, range(tot)), total=tot)
#                 )
#                 pool.close()
#             else:
#                 th_knns = [self.filter_by_th(i) for i in range(tot)]
#             return th_knns


# class knn_faiss(knn):
#     def __init__(
#         self,
#         feats,
#         k,
#         nprobe=128,
#         omp_num_threads=None,
#         rebuild_index=True,
#         verbose=True,
#         **kwargs
#     ):
#         import faiss

#         if omp_num_threads is not None:
#             faiss.omp_set_num_threads(omp_num_threads)
#         self.verbose = verbose
#         with Timer("[faiss] build index", verbose):
#             feats = feats.astype("float32")
#             size, dim = feats.shape
#             # build the index Inner Product (base on dimension), return byte * d vectors (d is dimension)
#             # normalize the vectors => inner product 
#             index = faiss.IndexFlatIP(dim) 
#             # add features to the index
#             index.add(feats)   
#         with Timer("[faiss] query topk {}".format(k), verbose):
#             # search the top k nearest neighbors
#             sims, nbrs = index.search(feats, k=k)
#             print(sims)
#             print(nbrs)
#             #sims = simaliarity, nbrs = index of the nearest neighbors
#             #distance = 1 - simliarity
#             #
#             self.knns = [
#                 (
#                     np.array(nbr, dtype=np.int32),
#                     1 - np.array(sim, dtype=np.float32),
#                 )
#                 for nbr, sim in zip(nbrs, sims)
#             ] 
#             # nbrs = [[1,2,...k], [3,4,...k], ...]
#             # sims = [[0.9,0.8,...k], [0.8,0.7,...k]
import pickle
import faulthandler
faulthandler.enable()
import faiss
import numpy as np
import torch

from dataset import LanderDataset
from utils import (build_knns, build_next_level, decode, density,
                   density_estimation, fast_knns2spmat, knns2ordered_nbrs,
                   l2norm, row_normalize, sparse_mx_to_indices_values)

data_path = 'handcrawl_data/train_encodings.pickle'
with open(data_path, "rb") as f:
    features, labels = pickle.load(f)
# print(features)
# print(labels)
k_list = [7] # giảm dần k_list
lvl_list = [2] # tăng dần lvl_list
gs = [] #các graph g được tạo từ các level khác nhau
nbrs = [] #các neighbor của các level khác nhau
ks = [] #các k là số lượng neighbor của các level khác nhau
faiss_gpu = 'faiss'
for k, l in zip(k_list, lvl_list): #k=10, l=1
    dataset = LanderDataset(
        features=features,
        labels=labels,
        k=k,
        levels=l,
    ) #tạo dataset từ các level khác nhau
    gs += [g for g in dataset.gs] #các graph g được tạo từ các level khác nhau
    ks += [k for g in dataset.gs] #[k for g in dataset.gs] 
    #print(ks)
    #các k là số lượng neighbor của các level khác nhau
    nbrs += [nbr for nbr in dataset.nbrs] #các neibor của các level khác nhau

print("Dataset Prepared.")

# k = 5
# features = l2norm(features)
# feats = features.astype("float32")
# global_features = features.copy()
# cluster_features = features
# global_num_nodes = features.shape[0] # Số lượng node
# global_edges = ([], []) #cạnh global
# global_peaks = np.array([], dtype=np.compat.long) # Đỉnh global
# ids = np.arange(global_num_nodes) # Tạo mảng từ 0 đến số lượng node
# size, dim = feats.shape
# # normalize the vectors => inner product 
# index = faiss.IndexFlatIP(dim) 
# # add features to the index
# index.add(feats)   

# sims, nbrs = index.search(feats, k=k)
# # print(sims)
# # print(nbrs)
# knns = [
#  (
#     np.array(nbr, dtype=np.int32),
#                     1 - np.array(sim, dtype=np.float32),
#                 )
#                 for nbr, sim in zip(nbrs, sims)
# ] 
# #print(knns)
# dists, nbrs = knns2ordered_nbrs(knns) #sắp xếp các k-NN theo thứ tự tăng dần dựa trên similiarity
# #print(dists)
# num, k_knn = dists.shape # (N,k)
# conf = np.ones((num,), dtype=np.float32) # (N,)
# ind_array = labels[nbrs] == np.expand_dims(labels, 1).repeat(k_knn, 1)
# #print(ind_array)
# pos = ((1 - dists[:, 1:]) * ind_array[:, 1:]).sum(1) # tổng các ( tích (1 - distance) với nhãn của các điểm )
# neg = ((1 - dists[:, 1:]) * (1 - ind_array[:, 1:])).sum(1) #tổng các ( tích (1- dist) với (1 - nhãn của các điểm )    )         
# #print(pos)
# #print(neg)    
# conf = (pos - neg) * conf # tính confidence
# conf /= k_knn - 1 # normalize
# dens = conf
# #print(dens)
# def build_graph( features, cluster_features, labels, density, knns,k):
#         import dgl
#         import torch
#         adj = fast_knns2spmat(knns, k) #create sparse matrix (ma trận kề)
#         adj, adj_row_sum = row_normalize(adj) #calculate row sum (tính tổng các hàng), đã chuẩn hóa tất cả các hàng
#         indices, values, shape = sparse_mx_to_indices_values(adj) #convert sparse matrix to indices, values, shape
#         print(values.shape)
#         g = dgl.graph((indices[1], indices[0])) #create graph from indices[1] (cột) and indices[0] (hàng)
#         g.ndata["features"] = torch.FloatTensor(features) #thêm features vào đồ thị
#         g.ndata["cluster_features"] = torch.FloatTensor(cluster_features) #thêm cluster_features vào đồ thị
#         g.ndata["labels"] = torch.LongTensor(labels) #thêm labels vào đồ thị
#         g.ndata["density"] = torch.FloatTensor(density) #thêm density vào đồ thị
#         g.edata["affine"] = torch.FloatTensor(values) #thêm giá trị sims trên ma trận kề vào đồ thị (đã chuẩn hoá)
#         # A Bipartite from  DGL sampler will not store global eid, so we explicitly save it here
#         g.edata["global_eid"] = g.edges(form="eid") #global edge id = edge id
#         g.ndata["norm"] = torch.FloatTensor(adj_row_sum) #thêm giá trị chuẩn hóa theo hàng vào đồ thị
#         g.apply_edges(
#             lambda edges: {
#                 "raw_affine": edges.data["affine"] / edges.dst["norm"]
#             }
#         ) #cập nhật giá trị cạnh raw_affine = giá trị cạnh (đã chuẩn hoá) / giá trị chuẩn hóa (tổng) của đỉnh đích
#         g.apply_edges(
#             lambda edges: {
#                 "labels_conn": (
#                     edges.src["labels"] == edges.dst["labels"]
#                 ).long()
#             }
#         )#cập nhật giá trị cạnh labels_conn = 1 nếu đỉnh nguồn và đỉnh đích có cùng label, ngược lại = 0
#         g.apply_edges(
#             lambda edges: {
#                 "mask_conn": (
#                     edges.src["density"] > edges.dst["density"]
#                 ).bool()
#             }
#         )#cập nhật giá trị cạnh mask_connection = True nếu đỉnh nguồn có giá trị mật độ lớn hơn đỉnh đích, ngược lại = False
#         return g
# g = build_graph(
#     features, cluster_features, labels, dens, knns, k)

# print('We have %d nodes.' % g.number_of_nodes())
# print('We have %d edges.' % g.number_of_edges())
# import networkx as nx
# import matplotlib.pyplot as plt
# # Since the actual graph is undirected, we convert it for visualization
# # purpose.
# nx_G = g.to_networkx().to_undirected()
# # Kamada-Kawaii layout usually looks pretty for arbitrary graphs
# pos = nx.kamada_kawai_layout(nx_G)
# nx.draw(nx_G, pos, with_labels=True, node_color=[[.7, .7, .7]])
# #nx.draw(nx_G, with_labels=True)

# plt.savefig('plotgraph.png', dpi=300, bbox_inches='tight')
# #plt.show()

#             # Decode peak nodes
# (
#                 new_pred_labels,
#                 peaks,
#                 global_edges,
#                 global_pred_labels,
#                 global_peaks,
#             ) = decode(
#                 g,
#                 0,
#                 "sim",
#                 True,
#                 ids,
#                 global_edges,
#                 global_num_nodes,
#                 global_peaks,
#             )
#             # Update labels peaks
# ids = ids[peaks]
#             # Build next level features
# features, labels, cluster_features = build_next_level(
#                 features,
#                 labels,
#                 peaks,
#                 global_features,
#                 global_pred_labels,
#                 global_peaks,
# )