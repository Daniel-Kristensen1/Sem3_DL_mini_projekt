
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, models, transforms
from torch.utils.data import DataLoader
from tqdm import tqdm
from pathlib import Path

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

DATA_PATH = Path(r"C:\Users\Daniel K\Desktop\DAKI\3. Semester\Deep Learning\miniProjekt\archive (1)\Banana Ripeness Classification Dataset")


NUM_CLASSES = 4

BATCH_SIZE = 32

NUM_EPOCHS = 10

LEARNING_RATE = 0.001

LOSS_FN = nn.CrossEntropyLoss()
