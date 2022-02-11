# Supervisor: Le Hoai Bac
# Author: Nguyen Thi Thu Hang, Tran Hoang Nam

# -------------------- Library -------------------- #
import os
import numpy as np
import argparse
import torch
from config import Config
# -------------------- Define function -------------------- #

# -------------------- Define parser -------------------- #
parser = argparse.ArgumentParser()
parser.add_argument('--start_iter', type=int, default=1, help='Start iteration (ex: 10001)')
parser.add_argument('--train_data_dir', default='', help='Training data directory')
parser.add_argument('--val_data_dir', default='', help='Validation data file')
parser.add_argument('--test_data_dir', default='', help='Testing data file')
parser.add_argument('--save_dir', default='', help='Trained model directory')
parser.add_argument('--embedded_method', default='', help='Embedded method')
parser.add_argument('--download_origin_image', type = bool, default=False, help='If you have origin data')
parser.add_argument('--download_edge_detected_image', type = bool, default=False, help='If you have edge detected data')
parser.add_argument('--download_processed_loader', type = bool, default=True, help='If you have data was converted to graph')

param = parser.parse_args()

# -------------------- Import config -------------------- #
config = Config()
config.save_dir = param.save_dir
config.train_data_dir = param.train_data_dir
config.val_data_dir = param.val_data_dir
config.test_data_dir = param.test_data_dir

# -------------------- Load data -------------------- #
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print(device)
print(config.train_data_dir)
print(config.test_data_dir)
data = list()
# Case 0: Using origin data
if (param.download_origin_image):
    print("original")
elif (param.download_edge_detected_image):
    print("download_edge_detected_image")
elif (param.download_processed_loader):
    print("download_processed_loader")
    #train_loader = torch.load(config.train_data_dir, map_location=device)
    test_loader = torch.load(config.test_data_dir, map_location=device)
