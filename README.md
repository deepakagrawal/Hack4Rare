# Hack4Rare

[![Open in Visual Studio Code](https://open.vscode.dev/badges/open-in-vscode.svg)](https://open.vscode.dev/organization/repository)

Please download the `pbta-gene-expression-kallisto.stranded.rds` data from *OpenPBTA* to `data/` directory. Instructions on how to download the dataset are [here](https://github.com/AlexsLemonade/OpenPBTA-analysis#data-access-via-cavatica).

How to run the code.

1. Run `src/create_raw_files.py` file to generate node and edge_list files.
2. Run `src/main.py` to generate the node embedding. This code is a based on [metapath2vec](https://ericdongyx.github.io/papers/KDD17-dong-chawla-swami-metapath2vec.pdf) paper. `pytorch_geometric` has incorporated the methodology [here](https://pytorch-geometric.readthedocs.io/en/latest/_modules/torch_geometric/nn/models/metapath2vec.html#MetaPath2Vec). `metapath2vec` is an approach based on unweighted edges. Our proposed approach generalizes this to weighted networks.
3. To visualize the generated embedding please use [projetor.tensorflow.org](http://projector.tensorflow.org/).
