import os
import random
import numpy as np
import cv2
from tqdm import tqdm
from glob import glob
import torch
from torch_geometric.data import Data
from sklearn.preprocessing import StandardScaler

class GraphDataset():
    def __init__(self, config):
        self.config = config
        self.data = list()

        if config.preloader is None:
            self.config.preloader = config.save_path+'/data.pt'
            self.edge_detection()
            self.make_graphs()
        else:
            self.data = torch.load(config.preloader, map_location=config.device)

    def edge_detection(self):
        # Here we use some very bad ways, will be replaced by a stable one later
        try:
            # run some command lines
            cmd1 = "LD_LIBRARY_PATH=/usr/local/lib && export LD_LIBRARY_PATH"
            cmd2 = "./edge_detection "+self.config.edge+" "+self.config.data_path
            status = os.system(cmd1 + "&&" + cmd2)
            if status != 0:
                raise Exception('Error! Let us try another way')
        except:
            from edge_detection import convert_edge_all_dir
            self.config.edge = 'prewitt'
            convert_edge_all_dir(self.config.data_path)
            
        self.edge_dir = self.config.data_path+'_'+self.config.edge

    def make_graphs(self):
        for class_id, subdir in enumerate(glob(self.edge_dir+'/*/')):
            print('Working on', subdir)
            for filename in tqdm(glob(subdir+'*')):
                self.data.append(self.image_to_graph(filename, class_id))

        random.shuffle(self.data)
        torch.save(self.data, self.config.preloader)

    @staticmethod
    def image_to_graph(filename, y):
        x = []
        edge = [[], []]

        img = cv2.imread(filename, cv2.IMREAD_GRAYSCALE)
        height, width = img.shape

        nodes = np.full((height, width), -1)
        nodeid = 0

        half = np.max(img)/2

        for i in range(height):
            for j in range(width):
                if img[i][j] >=half:
                    nodes[i][j] = nodeid
                    nodeid += 1
                    x.append((i, j))
                    # x.append([i, j, node[i][j]])

                    if j>0 and nodes[i][j-1] != -1:
                        edge[0].extend((nodes[i][j], nodes[i][j-1]))
                        edge[1].extend((nodes[i][j-1], nodes[i][j]))

                    if i>0 and nodes[i-1][j] != -1:
                        edge[0].extend((nodes[i][j], nodes[i-1][j]))
                        edge[1].extend((nodes[i-1][j], nodes[i][j]))

        scaler = StandardScaler()
        x = scaler.fit_transform(x)

        return Data(x = torch.tensor(x, dtype = torch.float),
                    edge_index = torch.tensor(edge, dtype = int),
                    y = torch.tensor([y]))

