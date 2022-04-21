import os
import random
from tqdm import tqdm
from glob import glob
from utils import graph_preparation
from pathlib import Path
import torch
from torch.utils.data import random_split
from torch_geometric.loader import DataLoader


class GraphDataset():
    def __init__(self, config):
        self.data = Path(config['PATH']['data'])
        self.save = Path(config['PATH']['save'])
        self.edge_detection = config['METHOD']['edge_detection']
        self.batch_size = int(config['HPARAMS']['batch_size'])

        if self.data.is_dir():
            self.dataset, self.labels = graph_preparation(self.data, self.save, edge_detection=self.edge_detection)
            random.shuffle(self.dataset)
            torch.save((self.dataset, self.labels), self.save / (self.data.name + '.pt'))
        else:
            self.dataset, self.labels = torch.load(self.data)

    @property
    def num_node_features(self):
        return self.dataset[0].num_node_features

    @property
    def n_classes(self):
        return len(self.labels)

    def getLoaders(self):
        dataset_size = len(self.dataset)
        train_set_size = int(dataset_size * 0.7)
        val_set_size = int(dataset_size * 0.15)
        test_set_size = dataset_size - train_set_size - val_set_size
        self.sizes = [train_set_size, val_set_size, test_set_size]

        train_set, val_set, test_set = random_split(self.dataset, self.sizes)

        train_loader = DataLoader(train_set, self.batch_size, shuffle = True, num_workers=4)
        val_loader = DataLoader(val_set, self.batch_size, shuffle = False, num_workers=4)
        test_loader = DataLoader(test_set, self.batch_size, shuffle = False, num_workers=4)

        return train_loader, val_loader, test_loader
