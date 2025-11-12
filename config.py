from pathlib import Path
"""

This file contains configuration parameters for the Deep learning model.

"""

# Paths
DATA_MAIN_FOLDER = Path(r"C:\Program Files (x86)\Programmering\3_semester\Mini_projekt\alpha_dataset\data_splits_ready")

TRAIN_JSON = DATA_MAIN_FOLDER / "train" / "train.json"
TEST_JSON = DATA_MAIN_FOLDER / "test" / "test.json"
VAL_JSON = DATA_MAIN_FOLDER / "val" / "val.json"

TRAIN_IMAGES = DATA_MAIN_FOLDER / "train" / "images"
TEST_IMAGES = DATA_MAIN_FOLDER / "test" / "images"
VAL_IMAGES = DATA_MAIN_FOLDER / "val" / "images"

# CLasses for classification
CLASSES = ["Adamant", "Clay", "Coal", "Copper", "Gold", "Iron", "Mined", "Mithril", "Motherload_ore", "Removable_ore", "Runeite", "Silver", "Tin"]
C = len(CLASSES)


# Model / Grid
IMAGE_SIZE = (448, 448) # (H, W)
S = 7 # Vores 14x14 grid
B = 1 # Antal bokse per, celle
D = B * 5 + C # Samlet dimensioner for en celle

# Hyperparametre
BATCH_SIZE = 64
EPOCHS = 135
LEARNING_RATE = 1e-4
""" Ved ik om vi vil have de her"""
WARMUP_EPOCHS  = 0
EPSILON        = 1e-6
WEIGHT_DECAY   = 0.0


# Loss function weights
COORD = 5.0
NOOBJ = 0.5

# Misc
SEED = 42
DEVICE = "cuda"