import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import balanced_accuracy_score, cohen_kappa_score,f1_score
from sklearn.linear_model import LogisticRegression
from pathlib import Path
import argparse
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)-8s %(message)s',
                    datefmt='%a, %d %b %Y %H:%M:%S', filename='data/classfication.log',
                    filemode='w')
consoleHandler = logging.StreamHandler()

logger = logging.getLogger()
logger.addHandler(consoleHandler)

parser = argparse.ArgumentParser()
parser.add_argument('--tflogs', help="Path of the tensorflow logs directory which has embeddings and metadata",
                    default='data/PBTA/4nd_typ_v1_embed100/tensorboard_logs/0000', type=Path)
parser.add_argument('--labels', default='pathology_diagnosis')
parser.add_argument('--output')
args = parser.parse_args()


solver='lbfgs'
multi_class='auto'

for node_type in ['sample', 'patient']:
    X = np.loadtxt(args.tflogs/node_type/'tensor.tsv', delimiter='\t')
    y = pd.read_csv(args.tflogs / node_type / 'metadata.tsv', sep='\t', usecols=[args.labels])
    y['grade_glioma'] = y.pathology_diagnosis.apply(lambda x: 'High-grade glioma' if 'High-grade glioma' in x else 'Low-grade glioma' if 'Low-grade glioma' in x else 'Other')
    train_x, test_x, train_y, test_y = train_test_split(X, y.grade_glioma, random_state=25, test_size=0.4, stratify=y.grade_glioma)
    clf = LogisticRegression(solver=solver, multi_class=multi_class, max_iter=500, n_jobs=-1, C=0.45).fit(train_x, train_y)
    train_acc = clf.score(train_x, train_y)
    test_acc = clf.score(test_x, test_y)
    logging.info(f'training accuracy for {node_type}: {train_acc}')
    logging.info(f'test accuracy for {node_type}: {test_acc}')
