import numpy as np
import cv2
import argparse
from tqdm import tqdm
from pathlib import Path
import torch
from torch_geometric.data import Data
from sklearn.preprocessing import StandardScaler
import sklearn.metrics as metrics

import matplotlib.pyplot as plt
import seaborn as sns

class Logger():
	def __init__(self, filename, message = None):
		self.file = open(filename, 'a')
		if message is not None:
			self.file.writelines("Message: "+message+'\n')

	def log(self, content, print_console = True):
		if not isinstance(content, str):
			content = str(content)
		self.file.writelines(content+'\n')
		if print_console:
			print(content)

	def __del__(self):
		self.file.close()

def to_pyg(x, edge_head, edge_tail, y):
	scaler = StandardScaler()
	x = scaler.fit_transform(x)

	return Data(x = torch.tensor(x, dtype = torch.float),
				edge_index = torch.tensor([edge_head, edge_tail], dtype = int),
				y = torch.tensor([y]))
	
def edge_detection(source, dest, method='prewitt', save_edge=False, lower_threshold=20, upper_threshold=300):
	if save_edge:
		edge_dir_path = dest / (source.name+'_'+method)
		edge_dir_path.mkdir(parents=True, exist_ok=True)

	k = 1 if method == 'prewitt' else 2
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
				w, h = img.shape
				img_edge = np.zeros(shape=(w, h), dtype=np.int16)
				img_edge = np.abs(img_x)+np.abs(img_y)

				nodes = np.full((h, w), -1)
				nodeid = 0

				x, edge_head, edge_tail = [], [], []
				for i in range(1, h-1):
					for j in range(1, w-1):
						if upper_threshold > img_edge[i, j] > lower_threshold:
							nodes[i, j] = nodeid
							nodeid += 1

							if nodes[i][j-1] != -1:
								edge_head += (nodes[i, j], nodes[i, j-1])
								edge_tail += (nodes[i, j-1], nodes[i, j])

							if nodes[i-1][j] != -1:
								edge_head += (nodes[i, j], nodes[i-1, j])
								edge_tail += (nodes[i-1, j], nodes[i, j])

							x.append((i, j, img[i, j], img_x[i, j], img_y[i, j]))
								
				dataset.append(to_pyg(x, edge_head, edge_tail, class_id))

				if save_edge:
					dest_dir_path = edge_dir_path / entry.name
					dest_dir_path.mkdir(parents=True, exist_ok=True)
					dest_file = dest_dir_path / source_file.name
					cv2.imwrite(str(dest_file), img_edge)

			class_id += 1
	return dataset, labels

def plot_confusion_matrix(cm, labels, save = None, title = 'Confusion matrix'):
	#assert len(cm) == len(labels)

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

def plot_ROC(cm, labels, save = None, title = 'ROC'):
	assert len(cm) == len(labels)
	# calculate the fpr and tpr for all thresholds of the classification
	fpr, tpr, threshold = metrics.roc_curve(cm, labels)
	roc_auc = metrics.auc(fpr, tpr)
	#display
	plt.title('Receiver Operating Characteristic (ROC)')
	plt.plot(fpr, tpr, 'b', label = 'AUC = %0.2f' % roc_auc)
	plt.legend(loc = 'lower right')
	plt.plot([0, 1], [0, 1],'r--')
	plt.xlim([0, 1])
	plt.ylim([0, 1])
	plt.ylabel('True Positive Rate')
	plt.xlabel('False Positive Rate')
	if save is not None:
		plt.savefig(save, bbox_inches='tight', format = save.name.split('.')[-1])
	else:
		plt.show()


if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument('task')
	parser.add_argument('source', type = Path)
	parser.add_argument('-d', '--dest', type = Path)
	args = parser.parse_args()

	if args.task == 'edge_detection':
		if args.dest is None:
			edge_detection(args.source, args.source.parent)
		else:
			edge_detection(args.source, args.dest)
	elif args.task == 'image_to_graph':
		# TODO
		pass
	elif args.task == 'raw_to_graphs':
		# TODO
		pass