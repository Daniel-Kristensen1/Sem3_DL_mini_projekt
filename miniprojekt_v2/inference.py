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


training_val = torch.load(weight_path, map_location="cpu")
weights = training_val["model_state_dict"]

##############
### Model ####
##############
print("\nSTEP 1: Initiate model with weights:" )
model = m.create_object_detector()
model.load_state_dict(weights)
model.to(config.DEVICE)
print(" Model loaded with trained weights.")
model.eval()

##############
# Inference ##
##############
print("\nSTEP 2: Run inference on image:" )
image_path = utils.get_image_path(image_dir_path, 0)

with torch.no_grad():
    outputs = model(utils.image_to_tensor(image_path))
print(" Inference sucessful." )

##############
# Draw boxes #
##############
print("\nSTEP 3: Draw bounding boxes on image:" )
print(" Drawing all bounding boxes...")
utils.show_all_bb_inf(outputs[0]["boxes"], image_path )
