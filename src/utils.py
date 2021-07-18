from typing import Optional, Callable, List, Union
import os.path as osp

import torch
import pandas
from torch_sparse import coalesce, transpose
from torch_geometric.data import (InMemoryDataset, Data, download_url,
                                  extract_zip, Dataset)
from sklearn import preprocessing


class pbta2vec(InMemoryDataset):
    r"""The heterogeneous PBTA dataset, consisting of nodes from
    type :obj:`"gene id"`, :obj:`"sample"` .

    Args:
        root (string): Root directory where the dataset should be saved.
        transform (callable, optional): A function/transform that takes in an
            :obj:`torch_geometric.data.Data` object and returns a transformed
            version. The data object will be transformed before every access.
            (default: :obj:`None`)
        pre_transform (callable, optional): A function/transform that takes in
            an :obj:`torch_geometric.data.Data` object and returns a
            transformed version. The data object will be transformed before
            being saved to disk. (default: :obj:`None`)
    """

    # url = 'https://www.dropbox.com/s/1bnz8r7mofx0osf/net_aminer.zip?dl=1'
    # y_url = 'https://www.dropbox.com/s/nkocx16rpl4ydde/label.zip?dl=1'

    def __init__(self, root: str, transform: Optional[Callable] = None,
                 pre_transform: Optional[Callable] = None,
                 use_node_attr: bool = False, use_edge_attr: bool = False,
                 cleaned: bool = False):
        super().__init__(root, transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0])
        if self.data.x is not None and not use_node_attr:
            num_node_attributes = self.num_node_attributes
            self.data.x = self.data.x[:, num_node_attributes:]
        if self.data.edge_attr is not None and not use_edge_attr:
            num_edge_attributes = self.num_edge_attributes
            # self.data.edge_attr = self.data.edge_attr[:, num_edge_attributes:]
            self.data.edge_attr = self.data.edge_attr


    @property
    def raw_file_names(self) -> List[str]:
        return [
            'id_gene.txt', 'id_transcript.txt', 'id_sample.txt',
            'transcript_gene.txt', 'sample_transcript.parquet'
        ]

    @property
    def processed_file_names(self) -> str:
        return 'data.pt'

    @property
    def num_node_attributes(self) -> int:
        if self.data.x is None:
            return 0
        return self.data.x.size(1)

    @property
    def num_edge_attributes(self) -> Union[int, dict]:
        if self.data.edge_attr is None:
            return 0
        if isinstance(self.data.edge_attr[0], dict):
            return {key: val.shape[1] for (key, val) in self.data.edge_weight[0].items()}
        else:
            return self.data.edge_attr.size(1)

    def download(self):
        pass

    def process(self):
        # Get sample labels.
        path = osp.join(self.raw_dir, 'id_sample.txt')
        sample = pandas.read_csv(path, sep='\t', index_col=1)
        sample.rename(columns={'id': 'idx'}, inplace=True)

        # Get gene labels.
        path = osp.join(self.raw_dir, 'id_gene.txt')
        gene = pandas.read_csv(path, sep='\t', names=['idx', 'name'], index_col=1)

        # Get transcript labels
        path = osp.join(self.raw_dir, 'id_transcript.txt')
        transcript = pandas.read_csv(path, sep='\t', names=['idx', 'name'], index_col=1)

        # Get sample<->transcript connectivity.
        path = osp.join(self.raw_dir, 'sample_transcript.parquet')
        sample_transcript = pandas.read_parquet(path)
        sample_transcript = torch.from_numpy(sample_transcript.values)
        sample_transcript = sample_transcript.t().contiguous()
        M, N = int(sample_transcript[0].max() + 1), int(sample_transcript[1].max() + 1)
        sample_transcript_index, sample_transcript_attr = coalesce(sample_transcript[:-1,:].long(), sample_transcript[-1,:], M, N)
        transcript_sample_index, transcript_sample_attr = transpose(sample_transcript_index, sample_transcript_attr, M, N)

        sample_transcript_attr = sample_transcript_attr.reshape([-1, 1])
        transcript_sample_attr = transcript_sample_attr.reshape([-1,1])

        # Get transcript<->gene connectivity.
        path = osp.join(self.raw_dir, 'transcript_gene.txt')
        transcript_gene = pandas.read_csv(path, sep='\t', header=None)
        transcript_gene = torch.from_numpy(transcript_gene.values)
        transcript_gene = transcript_gene.t().contiguous()
        M, N = int(transcript_gene[0].max() + 1), int(transcript_gene[1].max() + 1)
        transcript_gene, _ = coalesce(transcript_gene, None, M, N)
        gene_transcript, _ = transpose(transcript_gene, None, M, N)

        data = Data(
            edge_index_dict={
                ('gene', 'from', 'transcript'): gene_transcript,
                ('transcript', 'from', 'sample'): transcript_sample_index,
                ('sample', 'of', 'transcript'): sample_transcript_index,
                ('transcript', 'of', 'gene'): transcript_gene,
            },
            edge_weight={
                ('sample', 'of', 'transcript'): sample_transcript_attr,
                ('transcript', 'from', 'sample'): transcript_sample_attr,
            },
            node_dict={
                # 'gene': torch.from_numpy(gene['name'].values),
                'sample': torch.from_numpy(sample['short_hist_labels'].values),
                # 'transcript': torch.from_numpy(transcript['name'].values)
            },
            node_index_dict={
                'gene': torch.from_numpy(gene['idx'].values),
                'sample': torch.from_numpy(sample['idx'].values),
                'transcript': torch.from_numpy(transcript['idx'].values)
            },
            num_edges_dict={
                ('gene', 'from', 'transcript'): gene_transcript.shape[0],
                ('transcript', 'of', 'gene'): transcript_gene.shape[0],
                ('sample', 'of', 'transcript'): sample_transcript_index.shape[0],
                ('transcript', 'from', 'sample'): transcript_sample_index.shape[0],
            },
            num_nodes_dict={
                'sample': sample.shape[0],
                'transcript': transcript.shape[0],
                'gene': gene.shape[0],
            },
        )

        if self.pre_transform is not None:
            data = self.pre_transform(data)

        torch.save(self.collate([data]), self.processed_paths[0])


def cat(seq):
    seq = [item for item in seq if item is not None]
    seq = [item.unsqueeze(-1) if item.dim() == 1 else item for item in seq]
    return torch.cat(seq, dim=-1) if len(seq) > 0 else None