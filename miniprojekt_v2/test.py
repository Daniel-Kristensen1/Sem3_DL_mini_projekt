import modelV2 as m
import torch
import config
from torchmetrics.detection.mean_ap import MeanAveragePrecision
from torch.utils.data import DataLoader
from data_handler import DataHandler
import train

import modelV2 as m
import torch
import config
from torchmetrics.detection.mean_ap import MeanAveragePrecision
from torch.utils.data import DataLoader
from data_handler import DataHandler
import train

print("TESTING MODEL:")
print("\nSTEP 1: Load model weights.")
weight_path = config.WEIGHTS

weights = torch.load(weight_path, map_location=config.DEVICE)
weights = weights["model_state_dict"]

model = m.create_object_detector()
model.load_state_dict(weights)
model.to(config.DEVICE)
print(" Model loaded with trained weights.")

print("\nSTEP 2: Initiate test DataLoader: " )
test_dataset = DataHandler(
    json_path=config.TEST_JSON,
    images_dir=config.TEST_IMAGES,
    resize=config.IMAGE_SIZE,
    train=False
)

test_loader = DataLoader(
    test_dataset,
    batch_size=config.BATCH_SIZE,
    shuffle=False,
    num_workers=config.NUM_WORKERS,
    collate_fn=train.collate_fn
)
print(" Data loader initated.")

print("\nSTEP 3: Initiate metics - MeanAveragePrecision")
metric = MeanAveragePrecision() # Har brug for pred og target som input.
model.eval()

print(" Loop over test-set batches...")
for batch_idx, (images, targets) in enumerate(test_loader):
    print(f"\n Batch: {batch_idx+1}/{len(test_loader)}")
    

    print(" Get model predictions on test data...")
    # Henter model predictions til metrics. 
    images = [img.to(config.DEVICE) for img in images] #Flytter billede til GPU fra CPU
    with torch.no_grad():       
        preds = model(images) # Preds: dict(Boxes, scores, labels, Ground-truth boxes)
    

    preds = [{data_type: tensor_data.to(config.DEVICE) for data_type, tensor_data in p.items()} for p in preds] # Sender en liste af dictionaries(boxes, labels, scores) til GPU (hvis tilgængelig)
    targets = [{data_type: tensor_data.to(config.DEVICE) for data_type, tensor_data in t.items()} for t in targets] #Flytter alle tensors i dict'en til device(GPUen)
    print(" Update metrics with predictions and ground truth...")
    metric.update(preds, targets)

print(" Metrics update: Finished")

print("\nSTEP 4: Compute results.")
result = metric.compute()
print(result)
print("Final MeanAveragePrediction Results:")
lines = "---------------------------------------"
result.pop("classes", None)
[print(f"{data_type:15}: {output_tensor:.4f}") for data_type, output_tensor in result.items()] 

