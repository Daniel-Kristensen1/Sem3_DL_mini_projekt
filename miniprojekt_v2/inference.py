import modelV2 as m
import torch
import config
from torchmetrics.detection.mean_ap import MeanAveragePrecision
from torch.utils.data import DataLoader
from data_handler import DataHandler
import train
from pathlib import Path
import torchvision.transforms as T
from PIL import Image
import utils
weight_path = config.WEIGHTS

image_dir_path = config.TEST_IMAGES
image_data = config.TEST_JSON

# -------------------------------
# Load weights
# -------------------------------
training_val = torch.load(weight_path, map_location="cpu")
weights = training_val["model_state_dict"]

model = m.create_object_detector()
model.load_state_dict(weights)
model.to(config.DEVICE)
print("✅ Model loaded with trained weights.")
model.eval()

def preprocess_image(path):
    img = Image.open(path).convert("RGB")

    transform = T.Compose([
        T.Resize(config.IMAGE_SIZE),
        T.ToTensor()
    ])

    img_tensor = transform(img).unsqueeze(0)   # Add batch dimension
    return img_tensor

image_path = utils.get_image_path(0,image_dir_path)
image = preprocess_image(image_path)
image = image.to(config.DEVICE)  
with torch.no_grad():
    outputs = model(image)

print(outputs[0]["boxes"])
print(outputs[0]["labels"])
print(outputs[0]["scores"])

utils.show_all_bb_inf(outputs[0]["boxes"], image_path )