# Supervisor: Le Hoai Bac
# Author: Nguyen Thi Thu Hang, Tran Hoang Nam

# -------------------- Library -------------------- #
import os
import numpy as np
import argparse
import torch
from config import Config
from graphDataset import GraphDataset
from torch_geometric.loader import DataLoader
from model import GCN
# -------------------- Define function -------------------- #

# -------------------- Define parser -------------------- #
parser = argparse.ArgumentParser()
parser.add_argument('--data_path', default=None, help='Path to data directory')
parser.add_argument('--save_path', default=None, help='Directory path to save results')
parser.add_argument('--preloader', default=None, help='Processed data directory path to load')
parser.add_argument('--edge', default='prewitt', help='Edge detection method')
parser.add_argument('--embedding', default='graph_covid_net', help='Graph embedding method')

args = parser.parse_args()

# -------------------- Import config -------------------- #
config = Config(args)
config.show()
# -------------------- Load data -------------------- #
dataset = GraphDataset(config).data
split_index = (int)(config.train_ratio*len(dataset))
data_train = dataset[:split_index]
data_test = dataset[split_index:]

train_loader = DataLoader(data_train, batch_size=config.batch_size, shuffle = True)
test_loader = DataLoader(data_test, batch_size=config.batch_size, shuffle = False)
# -------------------- Train -------------------- #
model = GCN(input_channels = 2, hidden_channels=32, output_channels = config.n_classes)
model = model.to(config.device)
optimizer = torch.optim.Adam(model.parameters(), lr = 0.01)
criterion = torch.nn.CrossEntropyLoss(reduction = 'sum')

def train():
    model.train()

    for data in train_loader:  # Iterate in batches over the training dataset.
        out = model(data.x, data.edge_index, data.batch)  # Perform a single forward pass.
        loss = criterion(out, data.y)
        loss.backward()  # Derive gradients.
        optimizer.step()  # Update parameters based on gradients.
        optimizer.zero_grad()  # Clear gradients.

def test(loader):
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

for epoch in range(1, 40):
    train()
    train_acc, train_loss = test(train_loader)
    test_acc, test_loss = test(test_loader)
    print(f'Epoch: {epoch:03d}, Train Acc: {train_acc:.4f}, Train Loss: {train_loss:.4f}, Test Acc: {test_acc:.4f}, Test Loss: {test_loss:.4f}')