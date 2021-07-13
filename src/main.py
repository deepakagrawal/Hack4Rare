import os.path as osp
import numpy as np
import pandas as pd
import torch
from src.utils import pbta2vec
from src.metagene2vec import MetaGene2Vec
from pathlib import Path
import torch_geometric.transforms as T
# from torch_geometric.nn import MetaPath2Vec

# load the dataset
path = osp.join('data', 'PBTA')
MODEL_PATH = 'data/PBTA/torch_model_wt'
dataset = pbta2vec(path)
data = dataset[0]

print(type(data.edge_index_dict))
print(data.edge_index_dict[('gene', 'from', 'transcript')].shape)

print(type(data.num_nodes_dict))
print(data.num_nodes_dict)

device = 'cuda' if torch.cuda.is_available() else 'cpu'
# device = "cpu"

metapath = [
    ('gene', 'from', 'transcript'),
    ('transcript', 'from', 'sample'),
    ('sample', 'of', 'transcript'),
    ('transcript', 'of', 'gene')
]

model = MetaGene2Vec(data.edge_index_dict,
                     None,
                     embedding_dim=50,
                     metapath=metapath,
                     walk_length=50,
                     context_size=10,
                     walks_per_node=5,
                     num_negative_samples=5,
                     sparse=True
                    ).to(device)

loader = model.loader(batch_size=160, shuffle=True, num_workers=0)

for idx, (pos_rw, neg_rw) in enumerate(loader):
    if idx == 10: break
    print(idx, pos_rw.shape, neg_rw.shape)

print(pos_rw[0],neg_rw[0])

optimizer = torch.optim.SparseAdam(list(model.parameters()), lr=0.01)


if Path(MODEL_PATH).exists():
    checkpoint = torch.load(MODEL_PATH)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])


def train(epoch, log_steps=500, eval_steps=1000):
    model.train()

    total_loss = 0
    for i, (pos_rw, neg_rw) in enumerate(loader):
        optimizer.zero_grad()
        loss = model.loss(pos_rw.to(device), neg_rw.to(device))
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        if (i + 1) % log_steps == 0:
            print((f'Epoch: {epoch}, Step: {i + 1:05d}/{len(loader)}, '
                   f'Loss: {total_loss / log_steps:.4f}'))
            total_loss = 0

        # if (i + 1) % eval_steps == 0:
        #     acc = test()
        #     print((f'Epoch: {epoch}, Step: {i + 1:05d}/{len(loader)}, '
        #            f'Acc: {acc:.4f}'))


@torch.no_grad()
def test(train_ratio=0.6):
    model.eval()

    z = model('sample', batch=data.node_index_dict['sample'])
    y = data.node_index_dict['sample']

    perm = torch.randperm(z.size(0))
    train_perm = perm[:int(z.size(0) * train_ratio)]
    test_perm = perm[int(z.size(0) * train_ratio):]

    return model.test(z[train_perm], y[train_perm], z[test_perm],
                      y[test_perm], max_iter=100, n_jobs=-1)


for epoch in range(0):
    train(epoch)
    # acc = test(0.1)
    # print(f'Epoch: {epoch}, Accuracy: {acc:.8f}')


# torch.save({'epoch': epoch,
#             'model_state_dict': model.state_dict(),
#             'optimizer_state_dict': optimizer.state_dict()}, MODEL_PATH)

checkpoint = torch.load(MODEL_PATH)
model.load_state_dict(checkpoint['model_state_dict'])
model.cpu()


### generate checkpoint for tensorboard visualization


def get_embedding_metadata(node_type: str):
    z = model(node_type, batch=data.node_index_dict[node_type]).detach().numpy()
    np.savetxt(f'data/PBTA/unweighted_{node_type}_embedding.tsv', z, delimiter='\t')
    df = pd.read_csv(f'data/PBTA/raw/id_{node_type}.txt', names=['id', 'label'], sep='\t')
    # df = pd.DataFrame({'id': data.node_index_dict[node_type].numpy()})
    # df['label'] = 'dummy_label'
    df.to_csv(f'data/PBTA/unweighted_{node_type}_metadata.tsv', index=None, sep='\t')



get_embedding_metadata('transcript')
get_embedding_metadata('gene')




# get_embedding_metadata('sample')
