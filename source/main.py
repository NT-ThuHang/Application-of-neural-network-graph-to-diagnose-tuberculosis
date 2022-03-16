# Supervisor: Le Hoai Bac
# Author: Nguyen Thi Thu Hang, Tran Hoang Nam

# -------------------- Library -------------------- #
import os
import numpy as np
import argparse
from pathlib import Path
from tqdm import tqdm

from utils import Logger, plot_confusion_matrix
from config import Config
from graphDataset import GraphDataset
from model import GCN

import torch
from sklearn.metrics import classification_report, confusion_matrix
# -------------------- Define function -------------------- #

# -------------------- Define parser -------------------- #
parser = argparse.ArgumentParser()
parser.add_argument('data', help='Path to data directory, or processed data file', type = Path)
parser.add_argument('-s', '--save', default=None, help='Directory path to save intermediate results', type = Path)
parser.add_argument('--edge', default='prewitt', help='Edge detection method')
parser.add_argument('--embedding', default='_4_local', help='Graph embedding method')
parser.add_argument('--message')
parser.add_argument('--ckpt', help='Model checkpoint path')

args = parser.parse_args()

# -------------------- Import config -------------------- #
config = Config(args)
config.show()
# -------------------- Load data -------------------- #
dataset = GraphDataset(config)
train_loader, val_loader, test_loader = dataset.getLoaders()
print('Train-val-test sizes:', dataset.sizes)
# -------------------- Train -------------------- #
if args.ckpt:
    model = torch.load(args.ckpt, map_location=config.device)
else:
    model = GCN(input_channels = 5, hidden_channels=64, output_channels = dataset.n_classes)
    model = model.to(config.device)

optimizer = torch.optim.Adam(model.parameters(), lr = 5e-3)
scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.9, verbose = True)
criterion = torch.nn.CrossEntropyLoss(reduction = 'sum')

logger = Logger(config.save / 'log.txt', args.message)

def train():
    model.train()
    for data in train_loader:
        data = data.to(config.device)
        out = model(data.x, data.edge_index, data.batch)
        loss = criterion(out, data.y)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

def eval(loader):
    model.eval()

    correct, loss = 0, 0
    with torch.no_grad():
        for data in loader:   
            data = data.to(config.device)   
            out = model(data.x, data.edge_index, data.batch)
            pred = out.argmax(dim=1)
            correct += int((pred == data.y).sum())
            loss += criterion(out, data.y)
    return correct / len(loader.dataset), loss/ len(loader.dataset)


def test(model, loader):
    model.eval()
    y_true, y_pred, loss = [], [], 0
    with torch.no_grad():
        for data in loader:   
            data = data.to(config.device)   
            out = model(data.x, data.edge_index, data.batch)
            pred = out.argmax(dim=1)
            y_pred += [e.item() for e in pred]
            y_true += [e.item() for e in data.y]
            loss += criterion(out, data.y)

    report = classification_report(y_pred, y_true)
    c_mat = confusion_matrix(y_pred, y_true)
    loss = loss.item()/len(y_pred)
    plot_confusion_matrix(c_mat, labels = dataset.labels, 
                        save = config.save / 'confusion_matrix.svg')

    logger.log(report)
    logger.log('Confusion matrix:')
    logger.log(c_mat)
    logger.log('Loss: '+str(loss))

def train_loop(checkpoint_path, max_epoch = 200, lr_step = None, patience = 4):
    best_loss = 99.99
    counter = 0

    for epoch in range(1, max_epoch+1):
        train()
        train_acc, train_loss = eval(train_loader)
        val_acc, val_loss = eval(val_loader)
        epoch_result = f'Epoch: {epoch:03d}, Train Acc: {train_acc:.4f}, Train Loss: {train_loss:.4f}, Val Acc: {val_acc:.4f}, Val Loss: {val_loss:.4f}'
        logger.log(epoch_result)

        if val_loss < best_loss:
            best_loss = val_loss
            counter = 0
            torch.save(model, checkpoint_path)
        else:
            counter += 1
            if counter == patience:
                print('Early stopping')
                break

        if lr_step is not None:
            if epoch % lr_step == 0:
                scheduler.step()

checkpoint_path = config.save / 'model_checkpoint.ckpt' 
train_loop(checkpoint_path)
best_model = torch.load(checkpoint_path, map_location=config.device)
test(best_model, test_loader)
