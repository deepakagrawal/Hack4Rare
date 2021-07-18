import os.path as osp
import numpy as np
import pandas as pd
import torch
from src.utils import pbta2vec
from src.metagene2vec import MetaGene2Vec
from pathlib import Path
from tqdm import tqdm
import argparse
import logging
import tensorflow as tf
import tensorboard as tb
tf.io.gfile = tb.compat.tensorflow_stub.io.gfile
from torch.utils.tensorboard import SummaryWriter


def data_processing(path):
    # load the dataset
    # MODEL_PATH = 'data/PBTA/torch_model_wt_v1'
    dataset = pbta2vec(path)
    data = dataset[0]
    return data


def train(epoch, log_steps=500, eval_steps=1000, writer=None):
    "Function to training node embedding"
    model.train()

    total_loss = 0
    batch_loss = 0
    for i, (pos_rw, neg_rw) in enumerate(loader):
        optimizer.zero_grad()
        loss = model.loss(pos_rw.to(device), neg_rw.to(device))
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        batch_loss += loss.item()
        if (i + 1) % log_steps == 0:
            print((f'Epoch: {epoch}, Step: {i + 1:05d}/{len(loader)}, '
                   f'Loss: {batch_loss / log_steps:.4f}'))
            batch_loss = 0
            # total_loss = 0

        if (i + 1) % eval_steps == 0:
            _, acc = test()
            print((f'Epoch: {epoch}, Step: {i + 1:05d}/{len(loader)}, '
                   f'Acc: {acc:.4f}'))

    return total_loss/len(loader)



@torch.no_grad()
def test(train_ratio=0.1):
    model.eval()

    z = model('sample', batch=data.node_index_dict['sample'])
    y = data.node_dict['sample']

    perm = torch.randperm(z.size(0))
    train_perm = perm[:int(z.size(0) * train_ratio)]
    test_perm = perm[int(z.size(0) * train_ratio):]

    return model.test(z[train_perm], y[train_perm], z[test_perm],
                      y[test_perm], max_iter=100, n_jobs=-1)


def get_embedding_metadata(node_type: str, path: str, embed_file):
    "Get node embeddings and metadata files"
    z = model(node_type, batch=data.node_index_dict[node_type]).detach().numpy()
    np.savetxt(osp.join(path, node_type + embed_file), z, delimiter='\t')


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--modelpath", help="File name of the pytorch model", default="torch_model.pt")
    parser.add_argument("--embed", help="File name of the all embedding file", default="weighted_embedding_v1.tsv")
    parser.add_argument("--meta", help="File name of the all metada file", default="weighted_metadata_v1.tsv")
    parser.add_argument("--use_weight", help="Enable weighted metapath2vec", default=False, action="store_true")
    parser.add_argument("--output", help="Path of the output directory", default="data/PBTA/")
    parser.add_argument("-e", "--epoch", help="Number of epochs for training", default=0, type=int)
    parser.add_argument("-b", "--batch", help="number of samples in each batch", default=4, type=int)
    parser.add_argument("--embed_dim", help="Node Embedding dimension", default=64, type=int)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)-8s %(message)s',
                    datefmt='%a, %d %b %Y %H:%M:%S', filename= 'data/metagene2vec.log',
                    filemode='w')

    consoleHandler = logging.StreamHandler()


    logger = logging.getLogger("PBTAapp_logger")
    logger.addHandler(consoleHandler)

    MODEL_PATH = osp.join(args.output, args.modelpath)
    writer = SummaryWriter(f"runs/{Path(args.modelpath).stem}")

    logger.info("Load PBTA kallisto dataset and save preprocessed data")
    data = data_processing(args.output)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    logger.info(f"Device used for model training: {device}")

    metapath = [
        ('gene', 'from', 'transcript'),
        ('transcript', 'from', 'sample'),
        ('sample', 'of', 'transcript'),
        ('transcript', 'of', 'gene')
    ]

    logger.info(f"Metapath for the project: {metapath}")

    logger.info('''Initialize Pytorch Model. 
                   Accuracy will be checked based on correct classification of sample nodes to classes from 
                   Short Histologies in the histology file.''')
    model = MetaGene2Vec(data.edge_index_dict,
                         data.edge_weight if args.use_weight else None,
                         embedding_dim=args.embed_dim,
                         metapath=metapath,
                         walk_length=15,
                         context_size=5,
                         walks_per_node=10,
                         num_negative_samples=5,
                         sparse=True
                         ).to(device)

    loader = model.loader(batch_size=args.batch, shuffle=True, num_workers=0)

    logger.info("Show some positive and negative samples")
    for idx, (pos_rw, neg_rw) in enumerate(loader):
        if idx == 5: break
        print(idx, pos_rw.shape, neg_rw.shape)
        print(pos_rw[0], neg_rw[0])

    logger.info("initialize SparseAdam optimizer")
    optimizer = torch.optim.SparseAdam(list(model.parameters()), lr=0.01)

    if Path(MODEL_PATH).exists():
        if args.epoch <= 0:
            logger.info("Running in Evaluation Mode")
        logger.info("Loading exisiting model")
        checkpoint = torch.load(MODEL_PATH)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        _, acc = test()
        logger.info(f'Epoch: {args.epoch}, Accuracy: {acc:.8f}')
    else:
        logger.info("Running in Training Mode")

    logger.info("Start model training")
    for epoch in tqdm(range(args.epoch)):
        total_loss = train(epoch, writer=writer)
        train_acc, test_acc = test()
        writer.add_scalar('training_loss', total_loss, epoch)
        writer.add_scalar('Accuracy/test', test_acc, epoch)
        writer.add_scalar('Accuracy/train', train_acc, epoch)
        logger.info(f'Epoch: {epoch}, Accuracy: {test_acc:.8f}')
        logger.info(f"Saving model checkpoint for epoch: {epoch}")
        torch.save({'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict()}, MODEL_PATH)
        for node_type in data.num_nodes_dict.keys():
            embeddings = model(node_type, batch=data.node_index_dict[node_type]).detach().cpu().numpy()
            writer.add_histogram(f'Embeddings/{node_type}', embeddings + epoch, epoch)

    logger.info("Load saved model")
    checkpoint = torch.load(MODEL_PATH)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.cpu()

    logger.info("Getting node embedding from saved model")
    get_embedding_metadata('transcript', args.output, args.embed)
    get_embedding_metadata('gene', args.output, args.embed)
    get_embedding_metadata('sample', args.output, args.embed)
    if writer is not None:
        writer.close()
