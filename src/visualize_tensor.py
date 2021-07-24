import numpy as np
import pandas as pd
import argparse
from tsnecuda import TSNE
from pathlib import Path
import plotly.express as px


for node_type in ['gene']:
    tensor_fname = f"data/PBTA/4nd_typ_v1/tensorboard_logs/0000/{node_type}/tensor.tsv"
    meta_fname = f"data/PBTA/unwt_meta_3nd_type_test/tensorboard_logs/0000/{node_type}/metadata.tsv"
    if Path(tensor_fname).exists():
        node_embed = np.loadtxt(fname=tensor_fname)
        tsne = TSNE(verbose=True, n_iter=20, n_components=2)
        x_embed = tsne.fit_transform(node_embed)
        df_meta = pd.read_csv(meta_fname, sep='\t')
        df = pd.DataFrame(columns=['x', 'y'], data=x_embed)
        if node_type == 'patient':
            df['label'] = df_meta['pathology_diagnosis'].values
            fig = px.scatter(df, x='x', y='y', color='label', hover_data=['label'])
        elif node_type == 'sample':
            df['label'] = df_meta['pathology_diagnosis'].values
            df = pd.concat([df, df_meta[['cancer_predispositions', 'primary_site', 'extent_of_tumor_resection']]],axis=1)
            fig = px.scatter(df, x='x', y='y', color='label', hover_data=['cancer_predispositions', 'primary_site', 'extent_of_tumor_resection'])
        else:
            fig = px.scatter(df, x='x', y='y', color='label', hover_data=['label'])
        fig.write_html(f"data/PBTA/4nd_typ_v1/tensorboard_logs/0000/{node_type}/tsne.html")