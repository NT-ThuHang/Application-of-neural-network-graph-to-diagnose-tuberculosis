import cv2
import os
import numpy as np
from tqdm import tqdm
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('data', default = 'sample_data')
args = parser.parse_args()

def clahe(img):
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    return clahe.apply(img)

def edge_detection(image, k):
    kernel_x = np.array([[-1, 0, 1], [-k, 0, k], [-1, 0, 1]])
    kernel_y = np.array([[-1, -k, -1], [0, 0, 0], [1, k, 1]])
    img_x = cv2.filter2D(image, -1, kernel_x)
    img_y = cv2.filter2D(image, -1, kernel_y)
    h, w = image.shape
    img_edge = np.zeros(shape=(h, w), dtype=np.int16)
    img_edge = np.abs(img_x)+np.abs(img_y)
    img_edge = np.minimum(img_edge, 255)
    return np.uint8(img_edge)

def transform(src, transformer=None, dst=None):
    if dst is None:
        dst = src+'_'+transformer.__name__
    if os.path.exists(dst):
        import shutil
        shutil.rmtree(dst)

    os.makedirs(dst)
    print(f'Transforming from {src} to {dst}')
    for img_file in tqdm(os.listdir(src)):
        img = cv2.imread(src+'/'+img_file, cv2.IMREAD_GRAYSCALE)
        if transformer is not None:
            img = transformer(img)
        cv2.imwrite(dst+'/'+img_file, img)

for subdir in os.listdir(args.data):
    transform(src=args.data+'/'+subdir, transformer=clahe, dst=args.data+'_clahe/'+subdir+'/')
