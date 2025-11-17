
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, models, transforms
from torch.utils.data import DataLoader
from tqdm import tqdm
from pathlib import Path

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Datalokation
TRAIN_JSON = Path(r"C:\Program Files (x86)\Programmering\3_semester\Mini_projekt2.0\data_splits_ready\train\train.json")
TEST_JSON = Path(r"C:\Program Files (x86)\Programmering\3_semester\Mini_projekt2.0\data_splits_ready\test\test.json")
VAL_JSON = Path(r"C:\Program Files (x86)\Programmering\3_semester\Mini_projekt2.0\data_splits_ready\val\val.json")

TRAIN_IMAGES = Path(r"C:\Program Files (x86)\Programmering\3_semester\Mini_projekt2.0\data_splits_ready\train\images")
TEST_IMAGES = Path(r"C:\Program Files (x86)\Programmering\3_semester\Mini_projekt2.0\data_splits_ready\test\images")
VAL_IMAGES = Path(r"C:\Program Files (x86)\Programmering\3_semester\Mini_projekt2.0\data_splits_ready\val\images")


# Klasserne vi har i datasættet i rigtig rækkefølge
CLASSES = [
    "Adamant", "Clay", "Coal", "Copper", "Gold", "Iron",
    "Mined", "Mithril", "Motherload_ore", "Removable_ore",
    "Runeite", "Silver", "Tin",
]
# Dictionary med klasserne og deres ID'er
CLASSES_WITH_ID = {name: i + 1 for i, name in enumerate(CLASSES)}  # 0 reserveret til background

NUM_CLASSES = len(CLASSES)

BATCH_SIZE = 2

NUM_EPOCHS = 10

LEARNING_RATE = 0.001

LOSS_FN = []

IMAGE_SIZE = (640, 640)

NUM_WORKERS = 0
