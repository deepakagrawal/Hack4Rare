import torch
import numpy as np
import numba
import torch_sparse

def naive_weighted_choices1(weights, rowptr, rowcount):
    start_index = rowptr
    end_index = rowptr + rowcount
    rand = torch.rand(rowptr.shape)
    result = []
    for i,j,th in zip(start_index, end_index, rand):
        wt_sample = weights[i:j]
        wt_sample = torch.cumsum(wt_sample, dim=0)
        total = wt_sample[-1]
        if total == 0:
            return None
        idx = torch.searchsorted(wt_sample, th * total)
        result.append(idx)
    return result


src = torch.load("C:/Users/agraw/PycharmProjects/Hack4Rare/data/src.pt")
subset = torch.load("C:/Users/agraw/PycharmProjects/Hack4Rare/data/subset.pt")

rowptr, col, wt = src.csr()
rowcount = src.storage.rowcount()

rowcount = rowcount[subset]
rowptr = rowptr[subset]


n_iter = 40
import time
st = time.time()
for i in range(n_iter):
    temp = torch.tensor(naive_weighted_choices1(wt, rowptr, rowcount))
print((time.time() - st)/n_iter)