import torch
from torch_sparse import SparseTensor
import time


# @torch.jit.script
def naive_weighted_choices_torch(weights, rowptr, rowcount, col):
    start_index = rowptr
    end_index = rowptr + rowcount
    rand = torch.rand(rowptr.shape)
    result = []
    for i,j, th in zip(start_index, end_index, rand):
        wt_sample = weights[i:j]
        wt_sample = torch.cumsum(wt_sample, dim=0)
        total = wt_sample[-1]
        if total == 0:
            return None
        idx = torch.searchsorted(wt_sample, th * total)
        result.append(col[i+idx])
    return result


def sample_sparse(src: SparseTensor, num_samples: int = 1) -> torch.Tensor:
    norm_adj = src / src.sum(1).reshape([-1, 1])
    M, N = src.sizes()
    rowptr, col, weight = norm_adj.csr()
    rand = torch.rand([M, num_samples]) + torch.arange(M).reshape([-1, 1])
    weight_cumsum = torch.cumsum(weight, dim=0)
    sample = torch.searchsorted(weight_cumsum, rand)
    return col[sample]

src = torch.load("C:/Users/agraw/PycharmProjects/Hack4Rare/data/src.pt")
subset = torch.load("C:/Users/agraw/PycharmProjects/Hack4Rare/data/subset.pt")

rowptr, col ,wt = src.csr()
rowcount = src.storage.rowcount()


n_iter = 20
st = time.time()
for i in range(n_iter):
    if subset is not None:
        rowcount1 = rowcount[subset]
        rowptr1 = rowptr[subset]
    temp = torch.tensor(naive_weighted_choices_torch(wt, rowptr1, rowcount1, col))
print((time.time() - st)/n_iter)
