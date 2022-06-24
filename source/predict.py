import argparse
import numpy as np
import cv2

import torch
from model import GNN

from utils import to_pyg
from torch_geometric.profile import timeit, get_data_size, get_model_size
from torch_geometric.transforms.remove_isolated_nodes import RemoveIsolatedNodes

parser = argparse.ArgumentParser()
parser.add_argument('model')
parser.add_argument('img')
args = parser.parse_args()

def graph_preparation_file(source, edge_detection='prewitt', save_edge=False, lower_threshold=20, upper_threshold=240):
	k = 1 if edge_detection == 'prewitt' else 2
	kernel_x = np.array([[-1, 0, 1], [-k, 0, k], [-1, 0, 1]])
	kernel_y = np.array([[-1, -k, -1], [0, 0, 0], [1, k, 1]])

	class_id = 0
	img = cv2.imread(source, cv2.IMREAD_GRAYSCALE).astype(np.int16)
	img_x = cv2.filter2D(img, -1, kernel_x)
	img_y = cv2.filter2D(img, -1, kernel_y)

	img_edge = np.sqrt(np.power(img_x.astype(np.float32), 2)+np.power(img_y.astype(np.float32), 2))
	img_edge = np.minimum(img_edge, 255)
	img_edge = np.uint8(img_edge)

	h, w = img.shape
	nodes = np.full((h, w), -1)
	nodeid = 0
	is_node = (upper_threshold > img_edge) & (img_edge > lower_threshold)
	x, edge_head, edge_tail = [], [], []

	for i in range(1, h-1):
		for j in range(1, w-1):
			if is_node[i, j]:
				nodes[i, j] = nodeid

				if j>1 and is_node[i, j-1]:
					edge_head += (nodeid, nodes[i, j-1])
					edge_tail += (nodes[i, j-1], nodeid)

				if i>1 and is_node[i-1, j]:
					edge_head += (nodeid, nodes[i-1, j])
					edge_tail += (nodes[i-1, j], nodeid)

				x.append((i, j, img[i, j], img_x[i, j], img_y[i, j]))
				nodeid += 1

	graph = to_pyg(x, edge_head, edge_tail, class_id)
	remover = RemoveIsolatedNodes()
	graph = remover(graph)

	if save_edge:
		cv2.imwrite('edge.png', img_edge)

	return graph

device = torch.device("cpu")

model = torch.load(args.model, map_location=device)
graph = graph_preparation_file(args.img).to(device)

print(f'Model size: {get_model_size(model)/1024:.2f}KB')
print(f'Data size: {get_data_size(graph)/1024:.2f}KB')

@timeit()
@torch.no_grad()
def predict(model, graph):
    return model(graph.x, graph.edge_index, batch=None)

logit, inference_time = predict(model, graph)
score = torch.sigmoid(logit).item()

prediction = 'Normal' if score <= 0.5 else 'Tuberculosis'
print(f'Inference time: {inference_time:.4f}s')
print(f'Score: {score:.4f}')
print(f'Prediction: {prediction}')