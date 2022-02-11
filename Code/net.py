import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Linear, ReLU


import numpy as np
import glob
import cv2
import random
import matplotlib.pyplot as plt
from tqdm.notebook import tqdm


class NET:

    def __init__(self, config):
        self.config = config
        self.N = config.N

