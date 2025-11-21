from pathlib import Path
import getpass

import torch
from torchvision.models.detection.anchor_utils import AnchorGenerator



DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

user = getpass.getuser()
# Datalokation

if user == "Daniel K":
    print("User is Daniel")
    TRAIN_JSON = Path(r"C:\Users\Daniel K\Desktop\data_splits_ready\train\train.json")
    TEST_JSON = Path(r"C:\Users\Daniel K\Desktop\data_splits_ready\test\test.json")
    VAL_JSON = Path(r"C:\Users\Daniel K\Desktop\data_splits_ready\val\val.json")

    TRAIN_IMAGES = Path(r"C:\Users\Daniel K\Desktop\data_splits_ready\train\images")
    TEST_IMAGES = Path(r"C:\Users\Daniel K\Desktop\data_splits_ready\test\images")    
    VAL_IMAGES = Path(r"C:\Users\Daniel K\Desktop\data_splits_ready\val\images")    
    
    WEIGHTS = Path(r"C:\Users\Daniel K\Desktop\best_model.pth")

else: 
    TRAIN_JSON = Path(r"C:\Program Files (x86)\Programmering\3_semester\Mini_projekt2.0\data_splits_ready\train\train.json")
    TEST_JSON = Path(r"C:\Program Files (x86)\Programmering\3_semester\Mini_projekt2.0\data_splits_ready\test\test.json")
    VAL_JSON = Path(r"C:\Program Files (x86)\Programmering\3_semester\Mini_projekt2.0\data_splits_ready\val\val.json")


    TRAIN_IMAGES = Path(r"C:\Program Files (x86)\Programmering\3_semester\Mini_projekt2.0\data_splits_ready\train\images")
    TEST_IMAGES = Path(r"C:\Program Files (x86)\Programmering\3_semester\Mini_projekt2.0\data_splits_ready\test\images")
    VAL_IMAGES = Path(r"C:\Program Files (x86)\Programmering\3_semester\Mini_projekt2.0\data_splits_ready\val\images")

    WEIGHTS = Path(r"C:\Users\alext\Desktop\mappe_weights_and_loss\kmeans_anchor_change\best_model.pth")

# Sti til trænede vægte:


# Klasserne vi har i datasættet i rigtig rækkefølge
CLASSES = [
    "Adamant", "Clay", "Coal", "Copper", "Gold", "Iron",
    "Mined", "Mithril", "Motherload_ore", "Removable_ore",
    "Runeite", "Silver", "Tin",
]


CLASS_COLORS = { 
        "Adamant":        (80, 120, 70),    # dull green
        "Clay":           (200, 170, 120),  # pale brown/beige
        "Coal":           (30, 30, 30),     # black
        "Copper":         (170,100,50),     # orange/brown 
        "Gold":           (212,175,55),     # gold
        "Iron":           (130,120,110),    # gray with tint of brown
        "Mined":          (120,120,120),    # dark grey
        "Mithril":        (110,150,200),    # pale blue
        "Motherload_ore": (150,120,60),     # golden brown
        "Removable_ore":  (140,140,140),    # light grey
        "Runeite":        (45, 75, 160),    # deep blue
        "Silver":         (200,200,210),    # light silver
        "Tin":            (170,170,150),    # grey with yellow tint
    }


# Dictionary med klasserne og deres ID'er
CLASSES_WITH_ID = {name: i + 1 for i, name in enumerate(CLASSES)}  # 0 reserveret til background

NUM_CLASSES = len(CLASSES)

BATCH_SIZE = 4

NUM_EPOCHS = 100

LEARNING_RATE = 0.001

LOSS_FN = []

IMAGE_SIZE = (640, 640)

if DEVICE == torch.device("cpu"):
    NUM_WORKERS = 0
else:
    NUM_WORKERS = 2

# Hvis ikke man selv definere sizes og ratios, så er defult det som er lige under. 
DEFAULT_ANCHOR = AnchorGenerator(
        sizes=((32, 64, 128, 256, 512),),
        aspect_ratios=((0.5, 1.0, 2.0),) # Skalering ag højde på anchor boks.
        )
# Her er en test af nye, lavet ud fra k-means på JSON datasættet
K_MEAN_OPTIMIZED_ANCHORS = AnchorGenerator(
        sizes=((20, 30, 50, 70, 100),),
        aspect_ratios=((0.5, 1.0, 1.5),) # Skalering ag højde på anchor boks.
        )