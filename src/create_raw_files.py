import pyreadr
import pandas as pd
import numpy as np


df: pd.DataFrame = pyreadr.read_r("data/pbta-gene-expression-kallisto.stranded.rds")[None]
# df = df.head(10000)
df.gene_id.drop_duplicates().reset_index().to_csv('data/PBTA/raw/id_gene.txt', sep='\t', header=False)
df.transcript_id.to_csv('data/PBTA/raw/id_transcript.txt', sep='\t', header=False)
samples = pd.DataFrame({'sample_id': df.columns.to_numpy()[2:]})
samples.to_csv('data/PBTA/raw/id_sample.txt', sep='\t', header=False)
samples = pd.read_csv('data/PBTA/raw/id_sample.txt', sep='\t', names=['id', 'sample_id'])

print("Start writing sample_transcript.parquet")
df_transcript_sample = df.drop(columns='gene_id')
transcripts = pd.read_csv('data/PBTA/raw/id_transcript.txt', sep='\t', names=['id', 'transcript_id'])
df_transcript_sample = df_transcript_sample.merge(transcripts, on=['transcript_id'])
df_transcript_sample.drop(columns='transcript_id', inplace=True)
df_transcript_sample.rename(columns={'id': 'transcript_id'}, inplace=True)
df_transcript_sample.set_index('transcript_id', inplace=True)
df_transcript_sample = df_transcript_sample.replace({0: np.nan})
df_transcript_sample.rename(columns=dict(zip(samples.sample_id, samples.id)), inplace=True)
df_transcript_sample = df_transcript_sample.stack().reset_index()
df_transcript_sample.rename(columns={'level_1': 'sample_id', 0: 'weight'}, inplace=True)
df_transcript_sample.dropna(how='any', inplace=True)
df_transcript_sample[['sample_id', 'transcript_id', 'weight']].to_parquet('data/PBTA/raw/sample_transcript.parquet', index=None)
print("finished writing sample_transcript.parquet")

print("Start writing transcript_gene.txt")
df_transcript_gene = df[['transcript_id', 'gene_id']]
transcripts = pd.read_csv('data/PBTA/raw/id_transcript.txt', sep='\t', names=['id', 'transcript_id'])
df_transcript_gene = df_transcript_gene.merge(transcripts, on=['transcript_id'])
df_transcript_gene.drop(columns='transcript_id', inplace=True)
df_transcript_gene.rename(columns={'id': 'transcript_id'}, inplace=True)
gene = pd.read_csv('data/PBTA/raw/id_gene.txt', sep='\t', names=['id', 'gene_id'])
df_transcript_gene = df_transcript_gene.merge(gene, on=['gene_id'])
df_transcript_gene.drop(columns='gene_id', inplace=True)
df_transcript_gene.rename(columns={'id': 'gene_id'}, inplace=True)
df_transcript_gene[['transcript_id', 'gene_id']].to_csv('data/PBTA/raw/transcript_gene.txt', sep='\t', header=False, index=False)
print("finished writing transcript_gene.txt")

