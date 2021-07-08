import os.path as osp
import numpy as np
import pandas as pd
import torch
from src.utils import pbta2vec
from src.metagene2vec import MetaGene2Vec
# from torch_geometric.nn import MetaPath2Vec

# load the dataset
path = osp.join('data', 'PBTA')
MODEL_PATH = 'data/torch_model'
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
                     embedding_dim=100,
                     metapath=metapath,
                     walk_length=5,
                     context_size=3,
                     walks_per_node=5,
                     num_negative_samples=2,
                     sparse=True
                    ).to(device)

loader = model.loader(batch_size=128, shuffle=True, num_workers=0)

for idx, (pos_rw, neg_rw) in enumerate(loader):
    if idx == 10: break
    print(idx, pos_rw.shape, neg_rw.shape)

print(pos_rw[0],neg_rw[0])

optimizer = torch.optim.SparseAdam(list(model.parameters()), lr=0.01)


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

        if (i + 1) % eval_steps == 0:
            acc = test()
            print((f'Epoch: {epoch}, Step: {i + 1:05d}/{len(loader)}, '
                   f'Acc: {acc:.4f}'))


@torch.no_grad()
def test(train_ratio=0.1):
    model.eval()

    z = model('gene', batch=data.node_index_dict['gene'])
    y = data.node_index_dict['gene']

    perm = torch.randperm(z.size(0))
    train_perm = perm[:int(z.size(0) * train_ratio)]
    test_perm = perm[int(z.size(0) * train_ratio):]

    return model.test(z[train_perm], y[train_perm], z[test_perm],
                      y[test_perm], max_iter=150)


for epoch in range(10):
    train(epoch)
    acc = test()
    print(f'Epoch: {epoch}, Accuracy: {acc:.4f}')


torch.save(model.state_dict(), MODEL_PATH)

model.load_state_dict(torch.load(MODEL_PATH))
model.cpu()
z_transcript = model('transcript', batch=data.node_index_dict['transcript']).detach().numpy()
z_gene = model('gene', batch=data.node_index_dict['gene']).detach().numpy()

np.savetxt('data/transcript_embedding.tsv', z_transcript, delimiter='\t')
df_venue = pd.DataFrame({'id': data.node_index_dict['transcript'].numpy()})
df_venue['label'] = 'dummy_label'
df_venue.to_csv('data/transcript_metadata.tsv', index=False, sep='\t')
np.savetxt('data/gene_embedding.tsv', z_gene, delimiter='\t')
df_auth = pd.DataFrame({'id': data.node_index_dict['gene'].numpy()})
df_auth['label'] = 'dummy_label'
df_auth.to_csv('data/gene_metadata.tsv', index=False, sep='\t')
