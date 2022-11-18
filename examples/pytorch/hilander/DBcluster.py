from utils import (
    build_knns,
    fast_knns2spmat,
    knns2ordered_nbrs,
    l2norm,
    row_normalize,
    sparse_mx_to_indices_values,
)

class DBcluster(object):
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
        self.nbrs = []
        self.dists = []
        features = l2norm(features.astype("float32"))

        if faiss_gpu:
            knns = build_knns(features, self.k, "faiss_gpu")
        else:
            knns = build_knns(features, self.k, "faiss")
        
        dists, nbrs = knns2ordered_nbrs(knns)
        self.nbrs.append(nbrs)
        self.dists.append(dists)
        adj = fast_knns2spmat(knns, self.k) #create sparse matrix (ma trận kề)
        #print(adj)
        adj, adj_row_sum = row_normalize(adj) #calculate row sum (tính tổng các hàng)
        #print(adj, adj_row_sum)
        indices, values, shape = sparse_mx_to_indices_values(adj) #convert sparse matrix to indices, values, shape
        #print(indices, values, shape)
        self.adj = adj
        self.adj_row_sum = adj_row_sum
        self.indices = indices
        self.values = values
        self.shape = shape