import torch
import torch.nn as nn
import torchvision
from torchvision import transforms, datasets
from torch.utils.data import DataLoader
import pytorch_lightning as pl
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from torchvision.models import resnet50, ResNet50_Weights

import model     

print("STEP 1: Importing and verifying libraries...")
print(f"- PyTorch version: {torch.__version__}")
print(f"- Torchvision version: {torchvision.__version__}")
print(f"- PyTorch Lightning version: {pl.__version__}")
print(f"- CUDA available: {torch.cuda.is_available()}")
print("✓ Libraries imported and verified!")

print("\nSTEP 2: Defining model architecture...")
model = model.CustomResModel()

print(model)


