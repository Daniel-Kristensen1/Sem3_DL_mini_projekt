
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, models, transforms
from torch.utils.data import DataLoader
from tqdm import tqdm
from pathlib import Path

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

DATA_PATH = []

CLASSES = [
    "Adamant", "Clay", "Coal", "Copper", "Gold", "Iron",
    "Mined", "Mithril", "Motherload_ore", "Removable_ore",
    "Runeite", "Silver", "Tin",
]
NUM_CLASSES = len(CLASSES)

BATCH_SIZE = 32

NUM_EPOCHS = 10

LEARNING_RATE = 0.001

LOSS_FN = []
