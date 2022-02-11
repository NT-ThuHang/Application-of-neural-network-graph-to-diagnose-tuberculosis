# -------------------- Library -------------------- #
import cv2
import torch
import numpy as np
from config import Config
import torch_geometric as pyg
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GCNConv, GINConv
from torch_geometric.nn import global_mean_pool

config = Config()

def normalize(arr):
    arr=np.array(arr)
    m=np.mean(arr)
    s=np.std(arr)
    # if(s==0):
    #     return arr-arr
    return (arr-m)/s

def image_to_graph(filename, label):
    x = []
    edge = [[], []]

    img = cv2.imread(filename, cv2.IMREAD_GRAYSCALE)
    height, width = img.shape

    nodes = np.full((height, width), -1)
    nodeid = 0

    half = np.max(img)/2

    for i in range(height):
        for j in range(width):
            if img[i][j] >= half:
                nodes[i][j] = nodeid
                nodeid += 1
                x.append([i, j])
                # x.append([i, j, node[i][j]])

                if j>0 and nodes[i][j-1] != -1:
                  edge[0].extend([nodes[i][j], nodes[i][j-1]])
                  edge[1].extend([nodes[i][j-1], nodes[i][j]])

                if i>0 and nodes[i-1][j] != -1:
                  edge[0].extend([nodes[i][j], nodes[i-1][j]])
                  edge[1].extend([nodes[i-1][j], nodes[i][j]])

    x = normalize(x)
    x = x.astype(np.float32)

    # one hot encode
    y = np.zeros((1, config.n_classes))
    y[0][label] = 1
    y = y.astype(np.float32)

    return pyg.data.Data(x=torch.tensor(x),
                         edge_index=torch.tensor(edge, dtype = int),
                         y=torch.tensor(y))