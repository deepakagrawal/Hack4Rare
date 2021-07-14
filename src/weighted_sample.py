import torch
import numpy as np
import numba

@numba.jit(fastmath=True)
def numba_sum(a):
    result = 0
    for i in a:
        result += i
    return result


@numba.njit(fastmath=True)
def naive_weighted_choices(weights, rowptr, rowcount) -> list:
    result = []
    for i,j in zip(rowptr, rowcount):
        wt_sample = weights[i:i + j]
        wt_sample = wt_sample / wt_sample.sum()

        total = 0
        idx = 0
        th = np.random.rand()
        while th > total and idx < wt_sample.shape[0]-1:
            total += wt_sample[idx]
            idx += 1
        result.append(idx)
    return result


src = torch.load("C:/Users/agraw/PycharmProjects/Hack4Rare/data/src.pt")
subset = torch.load("C:/Users/agraw/PycharmProjects/Hack4Rare/data/subset.pt")

rowptr, col, wt = src.csr()
rowcount = src.storage.rowcount()

rowcount = rowcount[subset]
rowptr = rowptr[subset]


a = rowptr.numpy()
b = rowcount.numpy()
wt = wt.numpy()

n_iter = 40
import time
st = time.time()
for i in range(n_iter):
    temp = torch.tensor(naive_weighted_choices(wt, a, b))
print((time.time() - st)/n_iter)