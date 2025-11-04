
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
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=7, stride=2, padding="same") #Måske kan pytorch håndtere "same" padding selv?
        # Conv. layer 2
        self.conv2 = nn.Conv2d(in_channels=64, out_channels=192, kernel_size=3, stride=1, padding="same")
        # Conv. layer 3
        self.conv3_1 = nn.Conv2d(in_channels=192/2, out_channels=128, kernel_size=1, stride=1, padding="same")
        self.conv3_2 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=1, padding="same")
        self.conv3_3 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=1, stride=1, padding="same")
        self.conv3_4 = nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, stride=1, padding="same")
        # Conv. layer 4
        self.conv4_1 = nn.Conv2d(in_channels=512/2, out_channels=256, kernel_size=1, stride=1, padding="same") #Dette lag er tilpasset maxpool der kommer fra tideligere lag. Ellers skal in channel være 512
        self.conv4_2 = nn.Conv2d(in_channels=256, out_channels=512, kernel_size=1, stride=1, padding="same")
        
        self.conv4_3 = nn.Conv2d(in_channels=512, out_channels=256, kernel_size=1, stride=1, padding="same")
        self.conv4_4 = nn.Conv2d(in_channels=256, out_channels=512, kernel_size=1, stride=1, padding="same")
        
        self.conv4_5 = nn.Conv2d(in_channels=512, out_channels=256, kernel_size=1, stride=1, padding="same")
        self.conv4_6 = nn.Conv2d(in_channels=256, out_channels=512, kernel_size=1, stride=1, padding="same")
        
        self.conv4_7 = nn.Conv2d(in_channels=512, out_channels=256, kernel_size=1, stride=1, padding="same")
        self.conv4_8 = nn.Conv2d(in_channels=256, out_channels=512, kernel_size=1, stride=1, padding="same")
        #--------        
        self.conv4_9 = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=1, stride=1, padding="same")
        self.conv4_10 = nn.Conv2d(in_channels=512, out_channels=1024, kernel_size=3, stride=1, padding="same")
        
        # Conv. layer 5
        # Conv. layer 6
        # Fully connected layer 1
        # Fully connected layer 2
        


    def forward(self, x):
        return x