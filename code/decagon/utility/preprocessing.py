import numpy as np
import scipy.sparse as sp


def get_sparse_mat(a2b, a2idx, b2idx):
    n = len(a2idx)
    m = len(b2idx)
    assoc = np.zeros((n, m))
    for a, b_assoc in a2b.iteritems():
        if a not in a2idx:
            continue
        for b in b_assoc:
            if b not in b2idx:
                continue
            assoc[a2idx[a], b2idx[b]] = 1.
    assoc = sp.coo_matrix(assoc)
    return assoc

def sparse_to_tuple(sparse_mx):
    if not sp.isspmatrix_coo(sparse_mx): # 判断是否是coo类型(存储稀疏矩阵的一种格式)
        sparse_mx = sparse_mx.tocoo()
    coords = np.vstack((sparse_mx.row, sparse_mx.col)).transpose() # vstack()按垂直方向(行顺序)堆叠数组构成一个新的数组。
    values = sparse_mx.data # 对角线上的值
    shape = sparse_mx.shape # 1512
    return coords, values, shape