"""
This file re-uses implementation from https://github.com/yl-1993/learn-to-cluster
"""
import numpy as np
import torch
from sklearn import mixture

import dgl

from .density import density_to_peaks, density_to_peaks_vectorize

__all__ = [
    "peaks_to_labels",
    "edge_to_connected_graph",
    "decode",
    "build_next_level",
]


def _find_parent(parent, u):
    idx = []
    # parent is a fixed point
    while u != parent[u]:
        idx.append(u)
        u = parent[u]
    for i in idx:
        parent[i] = u
    return u


def edge_to_connected_graph(edges, num):
    parent = list(range(num))
    for u, v in edges:
        p_u = _find_parent(parent, u)
        p_v = _find_parent(parent, v)
        parent[p_u] = p_v

    for i in range(num):
        parent[i] = _find_parent(parent, i)
    remap = {}
    uf = np.unique(np.array(parent))
    for i, f in enumerate(uf):
        remap[f] = i
    cluster_id = np.array([remap[f] for f in parent])
    return cluster_id


def peaks_to_edges(peaks, dist2peak, tau):
    edges = []
    for src in peaks:
        dsts = peaks[src]
        dists = dist2peak[src]
        for dst, dist in zip(dsts, dists):
            if src == dst or dist >= 1 - tau:
                continue
            edges.append([src, dst])
    return edges


def peaks_to_labels(peaks, dist2peak, tau, inst_num):
    edges = peaks_to_edges(peaks, dist2peak, tau)
    pred_labels = edge_to_connected_graph(edges, inst_num)
    return pred_labels, edges


def get_dists(g, nbrs, use_gt):
    k = nbrs.shape[1]
    src_id = nbrs[:, 1:].reshape(-1)
    dst_id = nbrs[:, 0].repeat(k - 1)
    eids = g.edge_ids(src_id, dst_id)
    if use_gt:
        new_dists = (
            (1 - g.edata["labels_edge"][eids]).reshape(-1, k - 1).float()
        )
    else:
        new_dists = g.edata["prob_conn"][eids, 0].reshape(-1, k - 1)
    ind = torch.argsort(new_dists, 1)
    offset = torch.LongTensor(
        (nbrs[:, 0] * (k - 1)).repeat(k - 1).reshape(-1, k - 1)
    ).to(g.device)
    ind = ind + offset
    nbrs = torch.LongTensor(nbrs).to(g.device)
    new_nbrs = torch.take(nbrs[:, 1:], ind)
    new_dists = torch.cat(
        [torch.zeros((new_dists.shape[0], 1)).to(g.device), new_dists], dim=1
    )
    new_nbrs = torch.cat(
        [torch.arange(new_nbrs.shape[0]).view(-1, 1).to(g.device), new_nbrs],
        dim=1,
    )
    return new_nbrs.cpu().detach().numpy(), new_dists.cpu().detach().numpy()


def get_edge_dist(g, threshold):
    if threshold == "prob":
        return g.edata["prob_conn"][:, 0]
    return 1 - g.edata["raw_affine"]


def tree_generation(ng):
    # import networkx as nx
    # import matplotlib.pyplot as plt
    # print('We have %d nodes.' % ng.number_of_nodes())
    # print('We have %d edges.' % ng.number_of_edges()) 
    # nx_G = ng.to_networkx().to_undirected()
    # pos = nx.kamada_kawai_layout(nx_G)
    # nx.draw(nx_G, pos, with_labels=True, node_color=[[.7, .7, .7]])
    # #nx.draw(nx_G, with_labels=True)
    # plt.savefig('1.png', dpi=300, bbox_inches='tight')


    ng.ndata["keep_eid"] = torch.zeros(ng.number_of_nodes()).long() - 1
    #print(torch.zeros(ng.number_of_nodes()).long())
    def message_func(edges):
        #print(edges.data["edge_dist"])
        #print(edges.data[dgl.EID])
        return {"mval": edges.data["edge_dist"], "meid": edges.data[dgl.EID]}
        #mval = edges.data["edge_dist"] (message value)
        #meid = edges.data[dgl.EID] (message edge id)

    def reduce_func(nodes):
        ind = torch.min(nodes.mailbox["mval"], dim=1)[1] #index of minimum value
        keep_eid = nodes.mailbox["meid"].gather(1, ind.view(-1, 1))#gather the mesage edge id of the minimum value
        return {"keep_eid": keep_eid[:, 0]}
        #

    node_order = dgl.traversal.topological_nodes_generator(ng)# lấy topological order của đồ thị  
    #print(node_order)
    ng.prop_nodes(node_order, message_func, reduce_func) # propogate message từ node đầu tiên đến node cuối cùng


    eids = ng.ndata["keep_eid"] # lấy edge id của các node cần keep
    eids = eids[eids > -1] # lấy các edge id > -1
    edges = ng.find_edges(eids) # lấy các cạnh của các edge id
    treeg = dgl.graph(edges, num_nodes=ng.number_of_nodes()) # tạo đồ thị cây từ các cạnh

    # print('We have %d nodes.' % treeg.number_of_nodes())
    # print('We have %d edges.' % treeg.number_of_edges()) 
    # nx_G = treeg.to_networkx().to_undirected()
    # pos = nx.kamada_kawai_layout(nx_G)
    # nx.draw(nx_G, pos, with_labels=True, node_color=[[.7, .7, .7]])
    # #nx.draw(nx_G, with_labels=True)
    # plt.savefig('2.png', dpi=300, bbox_inches='tight')
    return treeg


def peak_propogation(treeg):
    treeg.ndata["pred_labels"] = torch.zeros(treeg.number_of_nodes()).long() - 1
    #node data 'pred_labels' = số node của treeG 
    peaks = torch.where(treeg.in_degrees() == 0)[0].cpu().numpy()
    #peaks = node có in_degree = 0
    treeg.ndata["pred_labels"][peaks] = torch.arange(peaks.shape[0])

    def message_func(edges):
        return {"mlb": edges.src["pred_labels"]}

    def reduce_func(nodes):
        return {"pred_labels": nodes.mailbox["mlb"][:, 0]}

    node_order = dgl.traversal.topological_nodes_generator(treeg)
    treeg.prop_nodes(node_order, message_func, reduce_func)
    pred_labels = treeg.ndata["pred_labels"].cpu().numpy()
    return peaks, pred_labels


def decode(
    g,
    tau,
    threshold,
    use_gt,
    ids=None,
    global_edges=None,
    global_num_nodes=None,
    global_peaks=None,
):
    # Edge filtering with tau and density
    den_key = "density" if use_gt else "pred_den" #dùng density (nếu use_gt = True) để lọc cạnh
    g = g.local_var() #copy graph
    g.edata["edge_dist"] = get_edge_dist(g, threshold) #lấy edge_dist dựa trên 'prob' hoặc 'raw_affine'
    g.apply_edges(
        lambda edges: {
            "keep": (edges.src[den_key] > edges.dst[den_key]).long()
            * (edges.data["edge_dist"] < 1 - tau).long()
        }
    )#"keep" = 1 nếu density của src > density của dst và edge_dist < 1- 0 (tau)
    #dist = 1 tức cos_sims = 0 => không giống
    eids = torch.where(g.edata["keep"] == 0)[0] #lấy eids của các cạnh không thỏa mãn điều kiện trên
    ng = dgl.remove_edges(g, eids) #xóa các cạnh không thỏa mãn điều kiện trên
    #new graph ng from g
    # Tree generation
    ng.edata[dgl.EID] = torch.arange(ng.number_of_edges()) #thêm cột dgl.EID (edge ID) vào new graph (tương ứng với đồ thị đã bỏ cạnh)
    treeg = tree_generation(ng) #tạo treeg từ ng , dùng topological traversal.
    #
    
    # Label propogation
    peaks, pred_labels = peak_propogation(treeg) #tìm các peak và gán nhãn cho các peak

    if ids is None:
        return pred_labels, peaks

    # Merge with previous layers
    src, dst = treeg.edges() #lấy src và dst của các cạnh của treeg (new graph)
    new_global_edges = (
        global_edges[0] + ids[src.numpy()].tolist(),
        global_edges[1] + ids[dst.numpy()].tolist(),
    ) #tạo global_edges mới từ global_edges cũ và (src và dst) của treeg (new graph)
    global_treeg = dgl.graph(new_global_edges, num_nodes=global_num_nodes) #tạo global_treeg từ global_edges mới
    global_peaks, global_pred_labels = peak_propogation(global_treeg) #tìm global_peaks và gán nhãn cho global_peaks
    return (
        pred_labels,
        peaks,
        new_global_edges,
        global_pred_labels,
        global_peaks,
    )


def build_next_level(
    features, labels, peaks, global_features, global_pred_labels, global_peaks
):
    global_peak_to_label = global_pred_labels[global_peaks]
    global_label_to_peak = np.zeros_like(global_peak_to_label)
    for i, pl in enumerate(global_peak_to_label):
        global_label_to_peak[pl] = i
    cluster_ind = np.split(
        np.argsort(global_pred_labels),
        np.unique(np.sort(global_pred_labels), return_index=True)[1][1:],
    )
    cluster_features = np.zeros((len(peaks), global_features.shape[1]))
    for pi in range(len(peaks)):
        cluster_features[global_label_to_peak[pi], :] = np.mean(
            global_features[cluster_ind[pi], :], axis=0
        )
    features = features[peaks]
    labels = labels[peaks]
    return features, labels, cluster_features
