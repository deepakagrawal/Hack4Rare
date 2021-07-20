from dateutil.parser import parse
import pyreadr
import pandas as pd
import numpy as np
import argparse
from pathlib import Path
import logging
from sklearn import preprocessing


parser = argparse.ArgumentParser()
parser.add_argument("--input", help="Path of the kallisto rds file", default="data/pbta-gene-expression-kallisto.stranded.rds")
parser.add_argument("--output", help="path of the output location where raw files will be stored", default="data/PBTA/raw/", type=Path)
parser.add_argument("--gene_id", help="filename of the id_gene file", default="id_gene.txt")
parser.add_argument("--transcript_id", help="filename of the id_transcript file", default="id_transcript.txt")
parser.add_argument("--sample_id", help="filename of the id_sample file", default="id_sample.txt")
parser.add_argument("--sample_transcript", help="filename of the sample_transcript file", default="sample_transcript.parquet")
parser.add_argument("--transcript_gene", help="filename of the transcript_gene file", default="transcript_gene.txt")
parser.add_argument("--hist", help="histology filename", default="data/pbta-histologies.tsv")
parser.add_argument("--chop_label", help="File path of samples having HGAT which are studied by CHOPS", default="data/PBTA/sample_in_chop_analysis.txt")
args = parser.parse_args()

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)-8s %(message)s',
                    datefmt='%a, %d %b %Y %H:%M:%S', filename= 'data/metagene2vec.log',
                    filemode='w')

consoleHandler = logging.StreamHandler()


logger = logging.getLogger("OpenPBTA_PreProcessor")
logger.addHandler(consoleHandler)

chop_samples = np.loadtxt(args.chop_label, dtype=str)

logger.info("Read input rds data")
df: pd.DataFrame = pyreadr.read_r(args.input)[None]
df['wt_sum'] = df[df.columns[2:]].sum(axis=1)
df = df[df.wt_sum >= 1e-3]
# df = df.head(10000)

logger.info("Read PBTA histology file")
df_hist: pd.DataFrame = pd.read_csv(args.hist, sep="\t")

logger.info("Save node_ids")
df.gene_id.drop_duplicates().reset_index().to_csv(args.output / args.gene_id, sep='\t', header=False, index=False)
df.transcript_id.drop_duplicates().to_csv(args.output / args.transcript_id, sep='\t', header=False)
samples: pd.DataFrame = pd.DataFrame({'Kids_First_Biospecimen_ID': df.columns.to_numpy()[2:-1]})
samples = samples.merge(df_hist, on=['Kids_First_Biospecimen_ID'], how='left')
samples.drop(columns=['sample_id'], inplace=True)
samples.reset_index(inplace=True)
samples.rename(columns={'index': 'id', 'Kids_First_Biospecimen_ID': 'sample_id'}, inplace=True)
broad_hist_label_encoder = preprocessing.LabelEncoder()
samples['broad_hist_labels'] = broad_hist_label_encoder.fit_transform(samples.broad_histology)
short_hist_label_encoder = preprocessing.LabelEncoder()
samples['short_hist_labels'] = short_hist_label_encoder.fit_transform(samples.short_histology)
samples["HGAT_chop_label"] = 0
samples.loc[samples.sample_id.isin(chop_samples), "HGAT_chop_label"] = 1
samples.to_csv(args.output / args.sample_id, sep='\t', index=False)
samples = pd.read_csv(args.output / args.sample_id, sep='\t', usecols=['id', 'sample_id'])




logger.info("Start writing sample_transcript.parquet")
df_transcript_sample = df.drop(columns=['gene_id', 'wt_sum'])
transcripts = pd.read_csv(args.output / args.transcript_id, sep='\t', names=['id', 'transcript_id'])
df_transcript_sample = df_transcript_sample.merge(transcripts, on=['transcript_id'])
df_transcript_sample.drop(columns='transcript_id', inplace=True)
df_transcript_sample.rename(columns={'id': 'transcript_id'}, inplace=True)
df_transcript_sample.set_index('transcript_id', inplace=True)
# df_transcript_sample = df_transcript_sample.replace({0: np.nan})
df_transcript_sample.rename(columns=dict(zip(samples.sample_id, samples.id)), inplace=True)
df_transcript_sample = df_transcript_sample.stack().reset_index()
df_transcript_sample.rename(columns={'level_1': 'sample_id', 0: 'weight'}, inplace=True)
df_transcript_sample.dropna(how='any', inplace=True)
df_transcript_sample = df_transcript_sample[df_transcript_sample.weight > 1e-4]
df_transcript_sample[['sample_id', 'transcript_id', 'weight']].to_parquet(args.output / args.sample_transcript, index=None)
logger.info("finished writing sample_transcript.parquet")

logger.info("Start writing transcript_gene.txt")
df_transcript_gene = df[['transcript_id', 'gene_id']]
transcripts = pd.read_csv(args.output / args.transcript_id, sep='\t', names=['id', 'transcript_id'])
df_transcript_gene = df_transcript_gene.merge(transcripts, on=['transcript_id'])
df_transcript_gene.drop(columns='transcript_id', inplace=True)
df_transcript_gene.rename(columns={'id': 'transcript_id'}, inplace=True)
gene = pd.read_csv(args.output / args.gene_id, sep='\t', names=['id', 'gene_id'])
df_transcript_gene = df_transcript_gene.merge(gene, on=['gene_id'])
df_transcript_gene.drop(columns='gene_id', inplace=True)
df_transcript_gene.rename(columns={'id': 'gene_id'}, inplace=True)
df_transcript_gene[['transcript_id', 'gene_id']].to_csv(args.output / args.transcript_gene, sep='\t', header=False, index=False)
logger.info("finished writing transcript_gene.txt")


