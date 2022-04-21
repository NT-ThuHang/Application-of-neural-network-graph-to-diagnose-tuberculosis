# Supervisor: Le Hoai Bac
# Author: Nguyen Thi Thu Hang, Tran Hoang Nam

# -------------------- Library -------------------- #
import os
import numpy as np
import argparse
from pathlib import Path
from tqdm import tqdm
from configparser import ConfigParser, ExtendedInterpolation

from utils import Logger, plot_confusion_matrix, plot_ROC
from graphDataset import GraphDataset
from model import GCN, GCN_GMT, GCN_test, MyGraphUNet
from torch_geometric.nn.models import GraphUNet, GIN, DeepGCNLayer

import torch
from sklearn.metrics import classification_report, confusion_matrix


# -------------------- Define parser -------------------- #
parser = argparse.ArgumentParser()
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
    
setup_config(config)

# -------------------- Load data -------------------- #
dataset = GraphDataset(config)
train_loader, val_loader, test_loader = dataset.getLoaders()
print('Train-val-test sizes:', dataset.sizes)

# -------------------- Load model -------------------- #
device = torch.device('cuda:'+config['OTHER']['device'] if torch.cuda.is_available() else 'cpu')

if Path(config['PATH']['checkpoint']).exists():
    print('Resume Trainning')
    model = torch.load(config['PATH']['checkpoint'], map_location=torch.device(device))
else:
    input_channels = dataset.num_node_features
    hidden_channels = eval(config['HPARAMS']['hidden_channels'])
    output_channels = dataset.n_classes

    model = GCN(channels = [input_channels, *hidden_channels, output_channels])
    # model = GraphUNet(input_channels, hidden_channels[0], output_channels, depth = 3)
    model = model.to(device)

n_params = sum(p.numel() for p in model.parameters())
print('Total parameters:', n_params)

optimizer = torch.optim.AdamW(model.parameters(), lr = float(config['HPARAMS']['lr']))
criterion = torch.nn.CrossEntropyLoss(reduction = 'sum')
logger = Logger(config['PATH']['logs'], config['OTHER']['message'])

def train():
    model.train()
    pbar = tqdm(train_loader, leave=False)
    for data in pbar:
        data = data.to(device)
        out = model(data.x, data.edge_index, data.batch)
        loss = criterion(out, data.y)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

def eval(loader):
    model.eval()

    correct, loss = 0, 0
    with torch.no_grad():
        for data in tqdm(loader, leave=False):   
            data = data.to(device)   
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
            data = data.to(device)   
            out = model(data.x, data.edge_index, data.batch)
            pred = out.argmax(dim=1)
            y_pred += [e.item() for e in pred]
            y_true += [e.item() for e in data.y]
            loss += criterion(out, data.y)
    
    report = classification_report(y_true, y_pred, digits =4)
    c_mat = confusion_matrix(y_true, y_pred)
    avg_loss = loss.item()/len(y_pred)
    plot_confusion_matrix(c_mat, labels = dataset.labels, save = Path(config['PATH']['save']) / 'confusion_matrix.png')

    # plot_ROC(c_mat, labels = dataset.labels, 
    #                     save = config['HPARAMS'].save / 'ROC.svg')

    logger.log("Result in test set")
    logger.log(report)
    logger.log('Confusion matrix:')
    logger.log(c_mat)
    logger.log('Loss: '+str(avg_loss))

def train_loop(config):
    best_loss = 99.99
    counter = 0 

    if config['HPARAMS']['lr_step']:
        scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, float(config['HPARAMS']['gamma']), verbose = True)

    for epoch in range(1, int(config['HPARAMS']['max_epoch'])+1):
        train()
        #train_acc, train_loss = eval(train_loader)
        val_acc, val_loss = eval(val_loader)
        epoch_result = f'Epoch: {epoch:03d}, Val Acc: {val_acc:.4f}, Val Loss: {val_loss:.4f}'
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
train_loop(config)
with open(config['PATH']['config'], 'w') as configfile:
    config.write(configfile)

best_model = torch.load(config['PATH']['checkpoint'], map_location=device)
test(best_model, test_loader)

