import numpy as np
import cv2
import argparse
from tqdm import tqdm
from pathlib import Path
import torch
from torch_geometric.data import Data
from sklearn.preprocessing import StandardScaler

# applying filter on a single image
def apply_filter(file, filter):

	# print("reading file---> "+str(filename))
	img = cv2.imread(str(file), cv2.IMREAD_GRAYSCALE)

	# img=cv2.resize(img,(250,250))
	# comment out the above line if there is memory issue i.e. need to resize all images to smaller dim

	h, w = img.shape
	# print("shape: "+str(h)+" x "+str(w)+"\n")
	# define filters
	horizontal = filter
	vertical = np.transpose(filter)

	# define images with 0s
	newgradientImage = np.zeros((h, w))

	# offset by 1
	for i in range(1, h - 1):
		for j in range(1, w - 1):
			
			horizontalGrad = (horizontal[0, 0] * img[i - 1, j - 1]) + \
							(horizontal[0, 2] * img[i - 1, j + 1]) + \
							(horizontal[1, 0] * img[i, j - 1]) + \
							(horizontal[1, 2] * img[i, j + 1]) + \
							(horizontal[2, 0] * img[i + 1, j - 1]) + \
							(horizontal[2, 2] * img[i + 1, j + 1])

			verticalGrad = (vertical[0, 0] * img[i - 1, j - 1]) + \
							(vertical[0, 1] * img[i - 1, j]) + \
							(vertical[0, 2] * img[i - 1, j + 1]) + \
							(vertical[2, 0] * img[i + 1, j - 1]) + \
							(vertical[2, 1] * img[i + 1, j]) + \
							(vertical[2, 2] * img[i + 1, j + 1])

			# Edge Magnitude
			mag = np.sqrt(pow(horizontalGrad, 2.0) + pow(verticalGrad, 2.0))
			newgradientImage[i - 1, j - 1] = mag

	return newgradientImage

def to_pyg(x, edge, y):
	scaler = StandardScaler()
	x = scaler.fit_transform(x)

	return Data(x = torch.tensor(x, dtype = torch.float),
				edge_index = torch.tensor(edge, dtype = int),
				y = torch.tensor([y]))
	
def edge_detection(source, dest, method='prewitt'):
	edge_dir_path = dest / (source.name+'_'+method)
	edge_dir_path.mkdir(parents=True, exist_ok=True)

	for entry in source.iterdir():
		if entry.is_dir():
			dest_dir_path = edge_dir_path / entry.name
			dest_dir_path.mkdir(parents=True, exist_ok=True)
			print("\n\n---reading directory "+str(entry)+"---\n")

			for source_file in tqdm(entry.iterdir()):
				imagemat = apply_filter(source_file, np.array([[-1,0,1],[-1,0,1],[-1,0,1]]))
				dest_file = dest_dir_path / source_file.name
				cv2.imwrite(str(dest_file), imagemat)

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

	return to_pyg(x, edge, label)


def raw_to_graphs(raw_dir : Path):
	print('Running raw_to_graphs')

	x_path = raw_dir / 'node_features.txt'
	A_path = raw_dir / 'edges.txt'
	y_path = raw_dir / 'graph_features.txt'

	x_file = x_path.open('r')
	A_file = A_path.open('r')
	y_file = y_path.open('r')

	dataset = list()

	for graph_entry in tqdm(y_file):
		n_nodes, n_edges, label = graph_entry.split(',')
		n_nodes, n_edges, label = int(n_nodes), int(n_edges), int(label)

		x, A_head, A_tail = [], [], []
		for _ in range(n_nodes):
			x_coord, y_coord, pix = x_file.readline().split(',')
			x.append((int(x_coord), int(y_coord), int(pix)))

		for _ in range(n_edges):
			head, tail = A_file.readline().split(',')
			A_head.append(int(head))
			A_tail.append(int(tail))

		dataset.append(to_pyg(x, [A_head, A_tail], label))

	x_file.close()
	A_file.close()
	y_file.close()

	return dataset

def pixel_analyze(filename):
	img = cv2.imread(filename, cv2.IMREAD_GRAYSCALE)
	
	import matplotlib.pyplot as plt
	plt.hist(img.flatten(), bins = 256)
	plt.savefig('analyze_result.png')

if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument('--task')
	parser.add_argument('-s', '--source', type = Path)
	parser.add_argument('-d', '--dest', type = Path)
	args = parser.parse_args()

	if args.task == 'edge_detection':
		if args.dest is None:
			edge_detection(args.source, args.source.parent)
		else:
			edge_detection(args.source, args.dest)
	elif args.task == 'pixel_analyze':
		pixel_analyze(args.source)
	elif args.task == 'image_to_graph':
		# TODO
		pass
	elif args.task == 'raw_to_graphs':
		# TODO
		pass