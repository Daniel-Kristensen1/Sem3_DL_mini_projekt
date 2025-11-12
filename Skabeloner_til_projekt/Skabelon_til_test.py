import torch
from pathlib import Path
from PIL import Image
from torchvision.transforms.functional import to_tensor
from torchvision.models.detection import fasterrcnn_resnet50_fpn_v2, FasterRCNNPredictor, FasterRCNN_ResNet50_FPN_V2_Weights

CLASSES = [
    "Adamant", "Clay", "Coal", "Copper", "Gold", "Iron",
    "Mined", "Mithril", "Motherload_ore", "Removable_ore",
    "Runeite", "Silver", "Tin",
]

def build_model(num_classes: int):
    weights = FasterRCNN_ResNet50_FPN_V2_Weights.COCO_V1
    model = fasterrcnn_resnet50_fpn_v2(weights=weights)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FasterRCNNPredictor(in_features, num_classes)
    return model

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = build_model(len(CLASSES) + 1)
ckpt = torch.load("checkpoints/fasterrcnn_resnet50_last.pth", map_location=device)
model.load_state_dict(ckpt["model"])
model.to(device).eval()

# Peg på et testbillede
image_path = Path(r"C:\...\data_splits_ready\test\images\cap_....png")
image = Image.open(image_path).convert("RGB")

# (Valgfrit) resize billedet til samme størrelse som træning – modellen kan også klare variable størrelser
# Fra minimaliteten: vi går bare direkte til tensor
x = to_tensor(image).to(device).unsqueeze(0)

with torch.no_grad():
    preds = model(x)[0]  # dict: 'boxes', 'labels', 'scores'

# Print de top-5 detektioner
scores = preds["scores"].tolist()
boxes = preds["boxes"].tolist()
labels = preds["labels"].tolist()

for i in range(min(5, len(scores))):
    print(f"{i+1}: {CLASSES[labels[i]-1]} score={scores[i]:.3f} box={boxes[i]}")
