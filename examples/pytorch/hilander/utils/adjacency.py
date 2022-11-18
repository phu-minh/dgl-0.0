#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
This file re-uses implementation from https://github.com/yl-1993/learn-to-cluster
"""

import numpy as np
import scipy.sparse as sp
from scipy.sparse import coo_matrix


def row_normalize(mx):
    '''chuẩn hoá ma trận mx theo hàng'''
    """Row-normalize sparse matrix"""
    #print(mx)
    rowsum = np.array(mx.sum(1))
   # print(rowsum)
    #if rowsum <= 0, keep its previous value
    rowsum[rowsum <= 0] = 1
    r_inv = np.power(rowsum, -1).flatten()
    #print(r_inv)
    r_inv[np.isinf(r_inv)] = 0.0
   # print(r_inv)
    r_mat_inv = sp.diags(r_inv)
    #print(mx)
    #print(mx.toarray()[0])
    print('---------')
    print(r_mat_inv)    
    #print('---------')
    mx = r_mat_inv.dot(mx)
    #print(mx.toarray()[0])#.sum())
    #print(mx.shape)
    return mx, r_inv


def sparse_mx_to_indices_values(sparse_mx):
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    #print(sparse_mx.toarray()[0])
    #print(sparse_mx.row)
    #print(sparse_mx.col)
    indices = np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64)
    values = sparse_mx.data 
    #print(values)
    shape = np.array(sparse_mx.shape)
    return indices, values, shape
