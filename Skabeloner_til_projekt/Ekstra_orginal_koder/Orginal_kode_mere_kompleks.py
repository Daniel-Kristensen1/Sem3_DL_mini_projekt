#!/usr/bin/env python
import argparse
import json
import math
import time
from pathlib import Path

import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import functional as F
from torchvision.models.detection import fasterrcnn_resnet50_fpn_v2, FasterRCNN_ResNet50_FPN_V2_Weights
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

CLASSES = [
    "Adamant", "Clay", "Coal", "Copper", "Gold", "Iron",
    "Mined", "Mithril", "Motherload_ore", "Removable_ore",
    "Runeite", "Silver", "Tin",
]
CLASS_TO_ID = {name: idx + 1 for idx, name in enumerate(CLASSES)}  # +1 to reserve 0 for background


class RuneOresDataset(Dataset):
    def __init__(self, split_dir: Path, split_name: str, train: bool):
        self.split_dir = split_dir
        self.train = train
        self.images_dir = split_dir / "images"
        self.entries = json.loads((split_dir / f"{split_name}.json").read_text())
        self.transforms = DetectionTransform(train=train)

    def __len__(self):
        return len(self.entries)

    def __getitem__(self, idx):
        item = self.entries[idx]
        image_name = Path(item["image"]).name
        image = Image.open(self.images_dir / image_name).convert("RGB")

        boxes, labels, areas = [], [], []
        img_w, img_h = image.size
        for obj in item.get("label", []):
            cls = obj["rectanglelabels"][0]
            w = obj.get("original_width", img_w)
            h = obj.get("original_height", img_h)
            x_min = obj["x"] / 100.0 * w
            y_min = obj["y"] / 100.0 * h
            width = obj["width"] / 100.0 * w
            height = obj["height"] / 100.0 * h
            x_max = x_min + width
            y_max = y_min + height
            boxes.append([x_min, y_min, x_max, y_max])
            labels.append(CLASS_TO_ID[cls])
            areas.append(width * height)

        target = {
            "boxes": torch.as_tensor(boxes, dtype=torch.float32),
            "labels": torch.as_tensor(labels, dtype=torch.int64),
            "image_id": torch.tensor([item["id"]], dtype=torch.int64),
            "area": torch.as_tensor(areas, dtype=torch.float32),
            "iscrowd": torch.zeros(len(boxes), dtype=torch.int64),
        }

        return self.transforms(image, target)


class DetectionTransform:
    def __init__(self, train: bool, resize_size=(480, 480)):
        self.train = train
        self.resize_size = resize_size

    def __call__(self, image, target):
        # Resize image and boxes
        orig_w, orig_h = image.size
        image = F.resize(image, self.resize_size)
        new_w, new_h = self.resize_size

        # Skalér bounding boxes tilsvarende
        scale_x = new_w / orig_w
        scale_y = new_h / orig_h
        boxes = target["boxes"]
        boxes = boxes * torch.tensor([scale_x, scale_y, scale_x, scale_y])
        target["boxes"] = boxes

        image = F.to_tensor(image)

        if self.train and torch.rand(1) < 0.5:
            image = F.hflip(image)
            width = image.shape[-1]
            boxes = target["boxes"].clone()
            x_min = width - boxes[:, 2]
            x_max = width - boxes[:, 0]
            boxes[:, 0] = x_min
            boxes[:, 2] = x_max
            target["boxes"] = boxes

        return image, target


def collate_fn(batch):
    return tuple(zip(*batch))




def build_model(num_classes):
    weights = FasterRCNN_ResNet50_FPN_V2_Weights.COCO_V1
    model = fasterrcnn_resnet50_fpn_v2(weights=weights)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    return model


def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"using device: {device}")
    data_root = Path(args.data_root)
    train_ds = RuneOresDataset(data_root / "train", "train", train=True)
    val_ds = RuneOresDataset(data_root / "val", "val", train=False)

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=collate_fn,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collate_fn,
    )

    model = build_model(num_classes=len(CLASSES) + 1).to(device)
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(params, lr=args.lr, weight_decay=1e-4)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.step_size, gamma=0.1)

    scaler = torch.amp.GradScaler('cuda', enabled=args.amp and device.type == "cuda")
    best_val = math.inf
    args.output.mkdir(parents=True, exist_ok=True)

    for epoch in range(args.epochs):
        model.train()
        epoch_loss = 0.0
        start = time.time()
        for images, targets in train_loader:
            images = [img.to(device) for img in images]
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

            optimizer.zero_grad()
            with torch.amp.autocast('cuda', enabled=scaler.is_enabled()):
                loss_dict = model(images, targets)
                losses = sum(loss_dict.values())
            scaler.scale(losses).backward()
            scaler.step(optimizer)
            scaler.update()

            epoch_loss += losses.item()

        lr_scheduler.step()
        avg_train = epoch_loss / len(train_loader)

        val_loss = evaluate(model, val_loader, device)
        elapsed = time.time() - start
        print(f"Epoch {epoch+1:02d}/{args.epochs} - "
              f"train_loss={avg_train:.3f} val_loss={val_loss:.3f} ({elapsed:.1f}s)")

        if val_loss < best_val:
            best_val = val_loss
            ckpt_path = args.output / "fasterrcnn_resnet50_best.pth"
            torch.save({"model": model.state_dict(), "epoch": epoch + 1, "val_loss": val_loss}, ckpt_path)

    print("Training complete. Best val loss:", best_val)


def evaluate(model, loader, device):
    was_training = model.training
    model.train()
    total = 0.0
    with torch.no_grad():
        for images, targets in loader:
            images = [img.to(device) for img in images]
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
            loss_dict = model(images, targets)
            total += sum(loss_dict.values()).item()
    model.train(was_training)
    return total / len(loader)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, required=True,
                        help="Path to data_splits_ready folder")
    parser.add_argument("--output", type=Path, default=Path("./checkpoints"))
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--step-size", type=int, default=10)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--amp", action="store_true")
    args = parser.parse_args()
    train(args)
