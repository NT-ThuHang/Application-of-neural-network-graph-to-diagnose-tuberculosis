import os
import numpy as np
import cv2
from tqdm import tqdm
import torch
from torch_geometric.data import Data
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import auc, roc_curve

import matplotlib.pyplot as plt
import seaborn as sns
from torch_geometric.transforms.remove_isolated_nodes import RemoveIsolatedNodes

class Logger():
	def __init__(self, filename, message = None):
		self.file = open(filename, 'a')
		self.filename = filename
		if message:
			self.file.writelines("Message: "+message+'\n')

	def log(self, content, print_console = True):
		if not isinstance(content, str):
			content = str(content)
		self.file.writelines(content+'\n')
		if print_console:
			print(content)

	def __del__(self):
		self.file.close()
		if os.path.getsize(self.filename) == 0:
			os.remove(self.filename)

def to_pyg(x, edge_head, edge_tail, y):
	scaler = StandardScaler()
	x = scaler.fit_transform(x)

	return Data(x = torch.tensor(x, dtype = torch.float),
				edge_index = torch.tensor([edge_head, edge_tail], dtype = int),
				y = torch.tensor([y]))
	
def graph_preparation(source, dest, edge_detection, save_edge=True, lower_threshold=20, upper_threshold=240):
	if save_edge:
		edge_dir_path = dest / (source.name+'_'+edge_detection)
		edge_dir_path.mkdir(parents=True, exist_ok=True)

	k = 1 if edge_detection == 'prewitt' else 2
	kernel_x = np.array([[-1, 0, 1], [-k, 0, k], [-1, 0, 1]])
	kernel_y = np.array([[-1, -k, -1], [0, 0, 0], [1, k, 1]])

	class_id = 0
	dataset, labels = [], []
	for entry in source.iterdir():
		if entry.is_dir():
			labels.append(entry.name)
			print("Reading directory "+str(entry))

			for source_file in tqdm(entry.iterdir()):
				img = cv2.imread(str(source_file), cv2.IMREAD_GRAYSCALE).astype(np.int16)
				img_x = cv2.filter2D(img, -1, kernel_x)
				img_y = cv2.filter2D(img, -1, kernel_y)
				img_edge = np.sqrt(np.power(img_x.astype(np.float32), 2)+np.power(img_y.astype(np.float32), 2))
				img_edge = np.minimum(img_edge, 255)
				img_edge = np.uint8(img_edge)
				h, w = img.shape
				nodes = np.full(img.shape, -1)
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
				dataset.append(remover(graph))

				if save_edge:
					dest_dir_path = edge_dir_path / entry.name
					dest_dir_path.mkdir(parents=True, exist_ok=True)
					dest_file = dest_dir_path / source_file.name
					cv2.imwrite(str(dest_file), img_edge)

			class_id += 1
	return dataset, labels

def plot_confusion_matrix(cm, labels, save = None, title = 'Confusion matrix'):
	assert len(cm) == len(labels)
	plt.figure()
	ax = sns.heatmap(cm, annot = True, cmap = 'Blues', fmt='d')
	ax.set_title(title)
	
	ax.set_xlabel('Predicted Values')
	ax.set_ylabel('Actual Values ')

	ax.xaxis.set_ticklabels(labels, rotation=40)
	ax.yaxis.set_ticklabels(labels, rotation=0)

	if save is not None:
		plt.savefig(save, bbox_inches='tight', format = save.name.split('.')[-1])
	else:
		plt.show()

def plot_ROC(y_true, y_pred, save = None):
	fpr, tpr, _ = roc_curve(y_true, y_pred)
	roc_auc = auc(fpr, tpr)
	plt.figure()
	lw = 2
	plt.plot(fpr, tpr, color='darkorange',
			 lw=lw, label='ROC curve (area = %0.4f)' % roc_auc)
	plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
	plt.xlim([0.0, 1.0])
	plt.ylim([0.0, 1.05])
	plt.xlabel('False Positive Rate')
	plt.ylabel('True Positive Rate')
	plt.title('Receiver operating characteristic example')
	plt.legend(loc="lower right")
	if save is not None:
		plt.savefig(save, bbox_inches='tight', format = save.name.split('.')[-1])
	else:
		plt.show()