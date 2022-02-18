import os
import random
import numpy as np
from tqdm import tqdm
from utils import edge_detection, image_to_graph, raw_to_graphs
import torch

class GraphDataset():
    def __init__(self, config):
        self.data = config.data
        self.save = config.save
        self.edge = config.edge
        
        if self.data.is_dir():
            self.edge_detection()
            self.make_graphs()
        else:
            self.dataset = torch.load(self.data, map_location=config.device)

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

        if not raw.is_dir():
            edge = self.save / (self.data.name+'_'+self.edge)
            for class_id, subdir in enumerate(edge.iterdir()):
                for filename in tqdm(subdir.iterdir()):
                    self.dataset.append(image_to_graph(str(filename), class_id))
        else:
            self.dataset = raw_to_graphs(raw)

        random.shuffle(self.dataset)
        torch.save(self.dataset, self.save / 'data.pt')
