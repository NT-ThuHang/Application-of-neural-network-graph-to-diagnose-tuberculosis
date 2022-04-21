# Supervisor: Le Hoai Bac
# Author: Nguyen Thi Thu Hang, Tran Hoang Nam

# -------------------- Library -------------------- #
import os
import numpy as np
import argparse
import shutil
from pathlib import Path
from tqdm import tqdm

from utils import Logger, plot_confusion_matrix, plot_ROC
from config import Config
from graphDataset import GraphDataset
from model import GCN, GCN_GMT, GCN_test
import configparser

import torch
from sklearn.metrics import classification_report, confusion_matrix


# -------------------- Define parser -------------------- #
parser = argparse.ArgumentParser()
# parser.add_argument('data', help='Path to data directory, or processed data file', type = Path)
#parser.add_argument('--trained_path', help='Path to data directory that consist of trained model', type = Path)
# parser.add_argument('-s', '--save_path', default=None, help='Directory path to save intermediate results', type = Path)
# parser.add_argument('--edge', default='prewitt', help='Edge detection method')
# parser.add_argument('--embedding', default='_4_local', help='Graph embedding method')
# parser.add_argument('--message')
# parser.add_argument('--lr', default = 5e-3, help='Learning rate', type = float)
# parser.add_argument('--lr_step', default = 5, help='Learning rate step', type = float)
# parser.add_argument('--gamma', default = 0.7, help='Gamma', type = float)
# parser.add_argument('--device', default = '3')
parser.add_argument('config', default = 'config.ini')

def setup_config(config):

    if (config['DEFAULT']['checkpoint_path'] == ''):
        # generate to use if not provided
        root = Path(config['DEFAULT']['save_path'])
        counter = 0
        while True:
            counter += 1
            path = root / 'result{:02d}'.format(counter)
            # if directory is not exist or empty
            if not path.exists() or not list(path.iterdir()):
                path.mkdir(parents=True, exist_ok = True)
                config['DEFAULT']['save_path'] = str(path)
                break
    print('Intermediate results will be stored in', str(config['DEFAULT']['save_path']))


args = parser.parse_args()
config = configparser.ConfigParser()
config.read(args.config)



# -------------------- Import config -------------------- #
setup_config(config)

# -------------------- Load data -------------------- #
dataset = GraphDataset(config['DEFAULT'])
train_loader, val_loader, test_loader = dataset.getLoaders()
print('Train-val-test sizes:', dataset.sizes)

# -------------------- Define function -------------------- #
#set up device
device = torch.device('cuda:'+config['DEFAULT']['device'] if torch.cuda.is_available() else 'cpu')

if (config['DEFAULT']['checkpoint_path'] != ''):
    model = torch.load(config['DEFAULT']['checkpoint_path'], map_location=torch.device(device))
else:
    input_channels = int(config['DEFAULT']['input_channels'])
    hidden_channels = tuple([int(x) for x in config['DEFAULT']['hidden_channels'][1:-2].split(", ")])
    output_channels = dataset.n_classes

    # model = GCN_GMT(channels = [input_channels, *hidden_channels, output_channels])
    model = GCN_test(channels = [input_channels, *hidden_channels, output_channels])
    model = model.to(device)

n_params = sum(p.numel() for p in model.parameters())
print('Total parameters:', n_params)

config['DEFAULT']['checkpoint_path'] = os.path.join(Path(config['DEFAULT']['save_path']), 'model_checkpoint.ckpt' )
config['DEFAULT']['logs_path'] = os.path.join(Path(config['DEFAULT']['save_path']), 'log.txt' )
config['DEFAULT']['data'] = os.path.join(Path(config['DEFAULT']['save_path']), 'data.pt' )

optimizer = torch.optim.AdamW(model.parameters(), lr = float(config['DEFAULT']['lr']))
criterion = torch.nn.CrossEntropyLoss(reduction = 'sum')
logger = Logger(Path(config['DEFAULT']['logs_path']), config['DEFAULT']['message'])

def train():
    model.train()
    for data in tqdm(train_loader):
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
        for data in tqdm(loader):   
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
    #plot_confusion_matrix(c_mat, labels = dataset.labels, save = os.path.join(Path(config['DEFAULT']['save_path']), 'confusion_matrix.png'))

    # plot_ROC(c_mat, labels = dataset.labels, 
    #                     save = config['DEFAULT'].save / 'ROC.svg')

    logger.log("Result in test set")
    logger.log(report)
    logger.log('Confusion matrix:')
    logger.log(c_mat)
    logger.log('Loss: '+str(avg_loss))

def train_loop(config):
    best_loss = 99.99
    counter = 0 

    if config['DEFAULT']['lr_step'] is not None:
        scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, float(config['DEFAULT']['gamma']), verbose = True)

    for epoch in range(1, int(config['DEFAULT']['max_epoch'])+1):
        train()
        #train_acc, train_loss = eval(train_loader)
        val_acc, val_loss = eval(val_loader)
        epoch_result = f'Epoch: {epoch:03d}, Val Acc: {val_acc:.4f}, Val Loss: {val_loss:.4f}'
        logger.log(epoch_result)

        if val_loss < best_loss:
            best_loss = val_loss
            counter = 0
            torch.save(model, config['DEFAULT']['checkpoint_path'])
        else:
            counter += 1
            if counter == int(config['DEFAULT']['patience']):
                print('Early stopping')
                break

        if config['DEFAULT']['lr_step'] is not None:
            if epoch % int(config['DEFAULT']['lr_step']) == 0:
                scheduler.step()

    

# -------------------- Training -------------------- #
train_loop(config)
with open(os.path.join(config['DEFAULT']['save_path'], 'config.ini'), 'w') as configfile:
    config.write(configfile)

best_model = torch.load(config['DEFAULT']['checkpoint_path'], map_location=device)
test(best_model, test_loader)

