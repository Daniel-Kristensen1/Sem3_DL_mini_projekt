
import os
import random
from pathlib import Path
from typing import Tuple

import numpy as np
from PIL import Image
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
import torchvision.transforms as T

#Ovenstående imports er fra ChatGPT. Aner ikke hvilke jeg skal bruge... Det skal sorteres senere.

class YoloCNN(nn.Module):
    def __init__(self):
        super().__init__()
        # Conv. layer 1
        # Conv. layer 2
        # Conv. layer 3
        # Conv. layer 4
        # Conv. layer 5
        # Conv. layer 6
        # Fully connected layer 1
        # Fully connected layer 2
        


    def forward(self, x):
        return x