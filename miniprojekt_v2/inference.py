import random

import torch

import config
import modelV2 as m
import utils


weight_path = config.WEIGHTS

image_dir_path = config.TEST_IMAGES
image_data = config.TEST_JSON

# Sæt # i variablen under, for fjern den anden for at lave forudsigelse på specifikt billede
image_num = random.randint(0, 171)  
# image_num = 0

training_val = torch.load(weight_path, map_location=config.DEVICE)
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
image_path = utils.get_image_path(image_dir_path, image_num)

with torch.no_grad():
    outputs = model(utils.make_tensor(image_path))
print(" Inference sucessful." )

##############
# Draw boxes #
##############
print("\nSTEP 3: Draw bounding boxes on image:" )
print(" Drawing all bounding boxes...")
utils.show_all_bb_inf(image_path, outputs[0]["boxes"],outputs[0]["labels"], outputs[0]["scores"] )