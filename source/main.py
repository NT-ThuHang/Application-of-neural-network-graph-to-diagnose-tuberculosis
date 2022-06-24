# Supervisor: Le Hoai Bac
# Author: Nguyen Thi Thu Hang, Tran Hoang Nam

# -------------------- Library -------------------- #
import os
import numpy as np
import argparse
from pathlib import Path
from tqdm import tqdm
from configparser import ConfigParser, ExtendedInterpolation
import random

from utils import Logger, plot_confusion_matrix, plot_ROC
from graphDataset import GraphDataset
from model import GNN

import torch
import torch_geometric as pyg
from torch_geometric.profile import profileit, timeit
from sklearn.metrics import classification_report, confusion_matrix


# -------------------- Define parser -------------------- #
parser = argparse.ArgumentParser()
parser.add_argument('data', default = 'sample_data')
parser.add_argument('config', default = 'config.ini')
args = parser.parse_args()

# -------------------- Import config -------------------- #
config = ConfigParser(interpolation=ExtendedInterpolation())
config.read(args.config)

def setup_config(config):
    root = Path(config['PATH']['save'])
    counter = 0
    while True:
        counter += 1
        path = root / 'result{:02d}'.format(counter)
        # if directory is not exist or empty
        if not path.exists() or not list(path.iterdir()):
            path.mkdir(parents=True, exist_ok = True)
            config['PATH']['save'] = str(path)
            break
    
    print('Intermediate results will be stored in', str(path))

    if config['OTHER']['seed']:
        pyg.seed_everything(int(config['OTHER']['seed']))
    
setup_config(config)

# -------------------- Load data -------------------- #
dataset = GraphDataset(args.data, config)
train_loader, val_loader, test_loader = dataset.getLoaders()
print('Train-val-test sizes:', dataset.sizes)

# -------------------- Load model -------------------- #
device = torch.device('cuda:'+config['OTHER']['device'] if torch.cuda.is_available() else 'cpu')
def load_model(config, device):
    if Path(config['PATH']['checkpoint']).exists():
        print('Resume Training')
        model = torch.load(config['PATH']['checkpoint'])
    else:
        input_channels = dataset.num_node_features
        hidden_channels = eval(config['HPARAMS']['hidden_channels'])
        output_channels = 1

        model = GNN(input_channels, hidden_channels, output_channels)

    return model.to(device)

model = load_model(config, device)
n_params = sum(p.numel() for p in model.parameters())
print('Total parameters:', n_params)

optimizer = torch.optim.AdamW(model.parameters(), lr = float(config['HPARAMS']['lr']))
criterion = torch.nn.BCEWithLogitsLoss(reduction = 'sum')
logger = Logger(config['PATH']['logs'], config['OTHER']['message'])

def train():
    model.train()

    correct, total_loss = 0, 0
    for data in tqdm(train_loader, leave=False, desc='Training'):
        data = data.to(device)
        out = model(data.x, data.edge_index, data.batch)
        loss = criterion(out, data.y.to(torch.float32))
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        pred = (out >= 0)
        correct += int((pred == data.y).sum())
        total_loss += loss.item()

    return correct / len(train_loader.dataset), total_loss/ len(train_loader.dataset)

def eval(loader):
    model.eval()

    correct, loss = 0, 0
    with torch.no_grad():
        for data in tqdm(loader, leave=False, desc='Validating'):   
            data = data.to(device)   
            out = model(data.x, data.edge_index, data.batch)
            pred = (out >= 0)
            correct += int((pred == data.y).sum())
            loss += criterion(out, data.y.to(torch.float32))
    return correct / len(loader.dataset), loss/ len(loader.dataset)

def test(model, loader):
    model.eval()
    y_true, y_pred, loss = [], [], 0
    with torch.no_grad():
        for data in tqdm(loader, desc='Testing', leave=False):   
            data = data.to(device)   
            out = model(data.x, data.edge_index, data.batch)
            y_pred += [e.item() for e in out]
            y_true += [e.item() for e in data.y]
            loss += criterion(out, data.y.to(torch.float32))

    plot_ROC(y_true, y_pred, save = Path(config['PATH']['save']) / 'ROC.png')
    y_pred = [y >= 0 for y in y_pred]
    report = classification_report(y_true, y_pred, digits=4)
    c_mat = confusion_matrix(y_true, y_pred)
    avg_loss = loss.item()/len(y_pred)
    plot_confusion_matrix(c_mat, labels = dataset.labels, save = Path(config['PATH']['save']) / 'confusion_matrix.png')

    logger.log("Result in test set")
    logger.log(report)
    logger.log('Confusion matrix:')
    logger.log(c_mat)
    logger.log('Loss: '+str(avg_loss))

@profileit()
def train_loop(model, config):
    best_loss = 99.99
    counter = 0 

    if config['HPARAMS']['lr_step']:
        scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, float(config['HPARAMS']['gamma']), verbose = True)

    for epoch in range(1, int(config['HPARAMS']['max_epoch'])+1):
        train_acc, train_loss = train()
        val_acc, val_loss = eval(val_loader)
        epoch_result = f'Epoch: {epoch:03d}, Train Acc: {train_acc:.4f}, Train Loss: {train_loss:.4f}, Val Acc: {val_acc:.4f}, Val Loss: {val_loss:.4f}'
        logger.log(epoch_result)

        if val_loss < best_loss:
            best_loss = val_loss
            counter = 0
            torch.save(model, config['PATH']['checkpoint'])
        else:
            counter += 1
            if counter == int(config['HPARAMS']['patience']):
                print('Early stopping')
                break

        if config['HPARAMS']['lr_step']:
            if epoch % int(config['HPARAMS']['lr_step']) == 0:
                scheduler.step()

# -------------------- Training -------------------- #
stats = train_loop(model, config)
print(stats)
with open(config['PATH']['config'], 'w') as configfile:
    config.write(configfile)

best_model = torch.load(config['PATH']['checkpoint'], map_location=device)
test(model, test_loader)

