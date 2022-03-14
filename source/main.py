# Supervisor: Le Hoai Bac
# Author: Nguyen Thi Thu Hang, Tran Hoang Nam

# -------------------- Library -------------------- #
import os
import numpy as np
import argparse
from pathlib import Path
import torch
from config import Config
from graphDataset import GraphDataset
from torch_geometric.loader import DataLoader
from torch.utils.data import random_split
from model import GCN
from sklearn.metrics import classification_report, confusion_matrix
# -------------------- Define function -------------------- #

# -------------------- Define parser -------------------- #
parser = argparse.ArgumentParser()
parser.add_argument('data', help='Path to data directory, or processed data file', type = Path)
parser.add_argument('-s', '--save', default=None, help='Directory path to save intermediate results', type = Path)
parser.add_argument('--edge', default='prewitt', help='Edge detection method')
parser.add_argument('--embedding', default='graph_covid_net', help='Graph embedding method')
parser.add_argument('--message')

args = parser.parse_args()

# -------------------- Import config -------------------- #
config = Config(args)
config.show()
# -------------------- Load data -------------------- #
dataset = GraphDataset(config).dataset

train_set_size = int(len(dataset) * 0.7)
val_set_size = int(len(dataset) * 0.15)
test_set_size = len(dataset) - train_set_size - val_set_size
sizes = [train_set_size, val_set_size, test_set_size]
print('Train-val-test sizes:', sizes)

train_set, val_set, test_set = random_split(dataset, sizes)

train_loader = DataLoader(train_set, config.batch_size, shuffle = True, num_workers =2)
val_loader = DataLoader(val_set, config.batch_size, shuffle = False)
test_loader = DataLoader(test_set, config.batch_size, shuffle = False)
# -------------------- Train -------------------- #
model = GCN(input_channels = 3, hidden_channels=64, output_channels = config.n_classes)
model = model.to(config.device)
optimizer = torch.optim.Adam(model.parameters(), lr = 5e-3)
criterion = torch.nn.CrossEntropyLoss(reduction = 'sum')

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

log_file = open(config.save / 'log.txt', 'w')
if args.message is not None:
    log_file.writelines(args.message+'\n')

# test phase
for epoch in range(1, 10):
    train()
    train_acc, train_loss = eval(train_loader)
    val_acc, val_loss = eval(val_loader)
    epoch_result = f'Epoch: {epoch:03d}, Train Acc: {train_acc:.4f}, Train Loss: {train_loss:.4f}, Val Acc: {val_acc:.4f}, Val Loss: {val_loss:.4f}'
    log_file.writelines(epoch_result+'\n')
    print(epoch_result)

model.eval()
correct, loss = 0, 0
y_true, y_pred = [], []
with torch.no_grad():
    for data in test_loader:   
        data = data.to(config.device)   
        out = model(data.x, data.edge_index, data.batch)
        pred = out.argmax(dim=1)
        y_true += [e.item() for e in pred]
        y_pred += [e.item() for e in data.y]
        loss += criterion(out, data.y)

report = classification_report(y_pred, y_true)
c_mat = confusion_matrix(y_pred, y_true)
loss = loss.item()/len(y_pred)

print(report)
print(c_mat)
print('Loss:', loss)

log_file.writelines(report+'\n')
log_file.writelines(str(c_mat)+'\n')
log_file.writelines(str(loss)+'\n')