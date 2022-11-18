import pickle

import numpy as np
import torch
from utils import (
    build_knns,
    build_next_level,
    decode,
    density_estimation,
    fast_knns2spmat,
    knns2ordered_nbrs,
    l2norm,
    row_normalize,
    sparse_mx_to_indices_values,
)

import dgl


class LanderDataset(object):
    def __init__(
        self,
        features,
        labels,
        cluster_features=None,
        k=10,
        levels=1,
        faiss_gpu=False,
    ):
        self.k = k
        self.gs = []
        self.nbrs = []
        self.dists = []
        self.levels = levels

        # Initialize features and labels
        # Khởi tạo Features (F) và Labels
        features = l2norm(features.astype("float32")) #
        global_features = features.copy()
        if cluster_features is None:
            cluster_features = features
        global_num_nodes = features.shape[0] # Số lượng node
        global_edges = ([], []) #cạnh global
        global_peaks = np.array([], dtype=np.compat.long) # Đỉnh global ?? 
        ids = np.arange(global_num_nodes) # Tạo mảng từ 0 đến số lượng node

        # Recursive graph construction
        # Xây dựng đồ thị đệ quy
        for lvl in range(self.levels):
            #không đủ node để tìm k-NN
            if features.shape[0] <= self.k:
                self.levels = lvl
                break
            #dùng GPU để tìm k-NN
            if faiss_gpu:
                knns = build_knns(features, self.k, "faiss_gpu")
            else:
                knns = build_knns(features, self.k, "faiss")    
            dists, nbrs = knns2ordered_nbrs(knns) #sắp xếp các k-NN theo thứ tự giảm dần dựa trên similiarity
            self.nbrs.append(nbrs)
            self.dists.append(dists)
            density = density_estimation(dists, nbrs, labels) #ước tính mật độ 

            g = self._build_graph(
                features, cluster_features, labels, density, knns
            ) #tạo đồ thị9bv
            self.gs.append(g)#thêm đồ thị vào danh sách

            #nếu level hiện tại lớn hơn level mong muốn thì dừng
            if lvl >= self.levels - 1:
                break

            # Decode peak nodes
            (
                new_pred_labels,
                peaks,
                global_edges,
                global_pred_labels,
                global_peaks,
            ) = decode(
                g,
                0,
                "sim",
                True,
                ids,
                global_edges,
                global_num_nodes,
                global_peaks,
            )
            # Update labels peaks
            ids = ids[peaks]
            # Build next level features
            features, labels, cluster_features = build_next_level(
                features,
                labels,
                peaks,
                global_features,
                global_pred_labels,
                global_peaks,
            )

    def _build_graph(self, features, cluster_features, labels, density, knns):
        adj = fast_knns2spmat(knns, self.k) #create sparse matrix (ma trận kề)
        #print(adj)
        adj, adj_row_sum = row_normalize(adj) #calculate row sum (tính tổng các hàng)
        #print(adj, adj_row_sum)
        indices, values, shape = sparse_mx_to_indices_values(adj) #convert sparse matrix to indices, values, shape
        #print(indices, values, shape)
        g = dgl.graph((indices[1], indices[0])) #create graph from indices[1] (cột) and indices[0] (hàng)
        g.ndata["features"] = torch.FloatTensor(features) #thêm features vào đồ thị
        g.ndata["cluster_features"] = torch.FloatTensor(cluster_features) #thêm cluster_features vào đồ thị
        g.ndata["labels"] = torch.LongTensor(labels) #thêm labels vào đồ thị
        g.ndata["density"] = torch.FloatTensor(density) #thêm density vào đồ thị
        g.ndata["norm"] = torch.FloatTensor(adj_row_sum) #thêm giá trị chuẩn hóa theo hàng vào đồ thị
        #--------------------------------------------------------------
        g.edata["affine"] = torch.FloatTensor(values) #thêm giá trị a/(a+b+c) trên ma trận kề vào đồ thị
        # A Bipartite from  DGL sampler will not store global eid, so we explicitly save it here
        g.edata["global_eid"] = g.edges(form="eid")    
        g.apply_edges(
            lambda edges: {
                "raw_affine": edges.data["affine"] / edges.dst["norm"]
            }
        ) #cập nhật giá trị cạnh = giá trị cạnh / giá trị chuẩn hóa của đỉnh đích
        g.apply_edges(
            lambda edges: {
                "labels_conn": (
                    edges.src["labels"] == edges.dst["labels"]
                ).long()
            }
        )#cập nhật giá trị cạnh = 1 nếu đỉnh nguồn và đỉnh đích có cùng label, ngược lại = 0
        g.apply_edges(
            lambda edges: {
                "mask_conn": (
                    edges.src["density"] > edges.dst["density"]
                ).bool()
            }
        )#cập nhật giá trị cạnh mask_connection = True nếu đỉnh nguồn có mật độ lớn hơn đỉnh đích, ngược lại = False
        return g

    def __getitem__(self, index):
        assert index < len(self.gs)
        return self.gs[index]

    def __len__(self):
        return len(self.gs)
