import os
import random
from tqdm import tqdm
from glob import glob
from utils import edge_detection, image_to_graph, raw_to_graphs

import torch
from torch.utils.data import random_split
from torch_geometric.loader import DataLoader


class GraphDataset():
    def __init__(self, config):
        self.data = config.data
        self.save = config.save
        self.edge = config.edge
        self.batch_size = config.batch_size

        if self.data.is_dir():
            self.edge_detection()
            self.make_graphs()
        else:
            self.dataset, self.labels = torch.load(self.data)
            
    @property
    def n_classes(self):
        return len(self.labels)

    def edge_detection(self):
        # Here we use some very bad ways, will be replaced by a stable one later
        try:
            # run some command lines
            cmd1 = "LD_LIBRARY_PATH=/usr/local/lib && export LD_LIBRARY_PATH"
            cmd2 = "source/graph_preparation "+self.edge+" "+str(self.data)+" "+str(self.save)
            if os.system(cmd1 + "&&" + cmd2) != 0:
                raise Exception('Error! Let us try another way')
        except:
            edge_detection(self.data, self.save)

    def make_graphs(self):
        raw = self.save / 'raw'
        self.dataset = list()
        self.labels = list()

        if not raw.is_dir():
            edge = self.save / (self.data.name+'_'+self.edge)
            for class_id, subdir in enumerate(edge.iterdir()):
                self.labels.append(subdir.name)
                for filename in tqdm(subdir.iterdir()):
                    self.dataset.append(image_to_graph(str(filename), class_id))
        else:
            self.dataset, self.labels = raw_to_graphs(raw)

        random.shuffle(self.dataset)
        torch.save((self.dataset, self.labels), self.save / 'data.pt')

    def getLoaders(self):
        dataset_size = len(self.dataset)
        train_set_size = int(dataset_size * 0.7)
        val_set_size = int(dataset_size * 0.15)
        test_set_size = dataset_size - train_set_size - val_set_size
        self.sizes = [train_set_size, val_set_size, test_set_size]

        train_set, val_set, test_set = random_split(self.dataset, self.sizes)

        train_loader = DataLoader(train_set, self.batch_size, shuffle = True, num_workers=2)
        val_loader = DataLoader(val_set, self.batch_size, shuffle = False, num_workers=2)
        test_loader = DataLoader(test_set, self.batch_size, shuffle = False, num_workers=2)

        return train_loader, val_loader, test_loader
