import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import argparse
from sklearn.manifold import TSNE
import umap
import umap.plot
from pathlib import Path
import plotly.express as px


for node_type in ['sample', 'patient']:
    tensor_fname = f"data/PBTA/4nd_typ_v1/tensorboard_logs/0000/{node_type}/tensor.tsv"
    meta_fname = f"data/PBTA/4nd_typ_v1/tensorboard_logs/0000/{node_type}/metadata.tsv"
    if Path(tensor_fname).exists():
        node_embed = np.loadtxt(fname=tensor_fname)
        reducer = umap.UMAP(n_epochs=50, n_neighbors=20, random_state=42, negative_sample_rate=50, force_approximation_algorithm=True, verbose=True)
        mapper = reducer.fit(node_embed)
        x_embed = reducer.transform(node_embed)
        # tsne = TSNE(verbose=2, n_jobs=-1, n_iter=2000, n_components=2, early_exaggeration=500, perplexity=50)
        # x_embed = tsne.fit_transform(node_embed)
        df_meta = pd.read_csv(meta_fname, sep='\t')
        df = pd.DataFrame(columns=['x', 'y'], data=x_embed)
        if node_type == 'patient':
            df = pd.concat([df, df_meta[['pathology_diagnosis', 'name']]], axis=1)
            # umap.plot.connectivity(mapper, edge_bundling='hammer', show_points=True, labels=df.label.values)
            # plt.savefig(f"data/PBTA/4nd_typ_v1/tensorboard_logs/0000/{node_type}/tsne_conn_hamm.pdf")
            fig = px.scatter(df, x='x', y='y', color='pathology_diagnosis', hover_data=['name', 'pathology_diagnosis'], color_discrete_sequence=px.colors.qualitative.Alphabet)
        elif node_type == 'sample':
            df = pd.concat([df, df_meta[['cancer_predispositions', 'primary_site', 'extent_of_tumor_resection', 'pathology_diagnosis', 'sample_id']]],axis=1)
            # umap.plot.connectivity(mapper, edge_bundling='hammer', show_points=True, labels=df['pathology_diagnosis'].values)
            # plt.savefig(f"data/PBTA/4nd_typ_v1/tensorboard_logs/0000/{node_type}/tsne_conn_hamm.pdf")
            fig = px.scatter(df, x='x', y='y', color='pathology_diagnosis', hover_data=['sample_id','pathology_diagnosis','cancer_predispositions', 'primary_site', 'extent_of_tumor_resection'])
        else:
            fig = px.scatter(df, x='x', y='y', color='label', hover_data=['label'])
        fig.write_html(f"data/PBTA/4nd_typ_v1/tensorboard_logs/0000/{node_type}/tsne.html")