import numpy as np
import cv2
import argparse
import os
from tqdm import tqdm
import glob
from pathlib import Path

# applying filter on a single image
def apply_filter(filename, filter):

	# print("reading file---> "+str(filename))
	img = cv2.imread(filename, cv2.IMREAD_GRAYSCALE)

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

# function for creating all edge-images under both covid and non-covid directories 
# since 2-class so COVID and NON-COVID present
def convert_edge_all_dir(source):
	dataset_dir_name = source.split('/')[-1]
	edge_dir_name = dataset_dir_name + '_prewitt'
	edge_dir_path = source[:-len(dataset_dir_name)]+edge_dir_name
  
	Path(edge_dir_path).mkdir(parents=True, exist_ok=True)
	
	for source_dir_path in glob.glob(source+'/*/'):
		source_dir_path = source_dir_path[:-1]
		source_dir_name = source_dir_path.split('/')[-1]
		dest_dir_path = edge_dir_path+'/'+source_dir_name
		Path(dest_dir_path).mkdir(parents=True, exist_ok=True)

		print("\n\n---reading directory "+source_dir_path+"---\n")

		for source_file_path in tqdm(glob.glob(source_dir_path+'/*')):
			imagemat = apply_filter(source_file_path, np.array([[-1,0,1],[-1,0,1],[-1,0,1]]))
			source_file_name = source_file_path.split('/')[-1]
			dest_file_path = dest_dir_path+'/'+source_file_name

			cv2.imwrite(dest_file_path, imagemat)

	print("\n---edge detection completed--\n")

if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument('--source')
	args = parser.parse_args()
	convert_edge_all_dir(args.source)

