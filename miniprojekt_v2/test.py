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


weight_path = config.WEIGHTS

weights = torch.load(weight_path, weights_only=True, map_location=config.DEVICE)
weights = weights["model_state_dict"]

model = m.create_object_detector()
model.load_state_dict(weights)
model.to(config.DEVICE)
print("✅ Model loaded with trained weights.")


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

print(f"✅ Test DataLoader ready: {len(test_dataset)} samples.")

# -------------------------------
# Initialize metric
# -------------------------------
metric = MeanAveragePrecision()
model.eval()

# -------------------------------
# Inference loop with print statements
# -------------------------------
for batch_idx, (images, targets) in enumerate(test_loader):
    print(f"\n➡️ Processing batch {batch_idx+1}/{len(test_loader)}")

    images = [img.to(config.DEVICE) for img in images]

    with torch.no_grad():
        preds = model(images)

    # Move predictions & targets to CPU
    preds = [{k: v.cpu() for k, v in p.items()} for p in preds]
    targets = [{k: v.cpu() for k, v in t.items()} for t in targets]

    # Print first sample predictions for sanity check
    print(f"Batch {batch_idx+1} first image predictions:")
    print(" Boxes:", preds[0]["boxes"])
    print(" Labels:", preds[0]["labels"])
    print(" Scores:", preds[0]["scores"])

    # Print target boxes for comparison
    print(" Ground-truth boxes:", targets[0]["boxes"])
    print(" Ground-truth labels:", targets[0]["labels"])

    # Update metric
    metric.update(preds, targets)

# -------------------------------
# Compute final metric
# -------------------------------
result = metric.compute()
print("\n🎯 Final evaluation metrics:")
lines = "---------------------------------------"
print(f"\nmap:                {result['map'].item()}")
print(lines)
print(f"map_50:             {result['map_50'].item()}")
print(lines)
print(f"map_75:             {result['map_75'].item()}")
print(lines)
print(f"map_small:          {result['map_small'].item()}")
print(lines)
print(f"map_medium:         {result['map_medium'].item()}")
print(lines)
print(f"map_large:          {result['map_large'].item()}")
print(lines)
print(f"mar_1:              {result['mar_1'].item()}")
print(lines)
print(f"mar_10:             {result['mar_10'].item()}")
print(lines)
print(f"mar_100:            {result['mar_100'].item()}")
print(lines)
print(f"mar_small:          {result['mar_small'].item()}")
print(lines)
print(f"mar_medium:         {result['mar_medium'].item()}")
print(lines)
print(f"mar_large:          {result['mar_large'].item()}")
print(lines)
print(f"map_per_class:      {result['map_per_class'].item()}")
print(lines)
print(f"mar_100_per_class:  {result['mar_100_per_class'].item()}")
print(lines)
print(f"classes:            {result['classes']}")
print(lines*2)

print(result)
