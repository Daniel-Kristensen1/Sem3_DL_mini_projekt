#!/usr/bin/env python
import argparse, json, time
from pathlib import Path

import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from torchvision.transforms import functional as F
from torchvision.models.detection import fasterrcnn_resnet50_fpn_v2, FasterRCNN_ResNet50_FPN_V2_Weights
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

# 1) Klassenavne: +1 til background håndteres i modellen (vi sætter num_classes = len(CLASSES)+1)
CLASSES = [
    "Adamant", "Clay", "Coal", "Copper", "Gold", "Iron",
    "Mined", "Mithril", "Motherload_ore", "Removable_ore",
    "Runeite", "Silver", "Tin",
]
CLASS_TO_ID = {name: i + 1 for i, name in enumerate(CLASSES)}  # 0 reserveret til background

# 2) Dataset: læser Label Studio JSON (brugte procenter), åbner billedet, laver boxes+labels
class RuneOresDataset(Dataset):
    def __init__(self, split_dir: Path, split_name: str, train: bool, resize=(640, 640)):
        self.images_dir = split_dir / "images"
        self.entries = json.loads((split_dir / f"{split_name}.json").read_text(encoding="utf-8"))
        self.train = train
        self.resize = resize  # (W, H)

    def __len__(self):
        return len(self.entries)

    def __getitem__(self, idx):
        item = self.entries[idx]
        image_name = Path(item["image"]).name
        image = Image.open(self.images_dir / image_name).convert("RGB")

        # Læs bokse fra Label Studio (procenter -> pixels)
        boxes, labels = [], []
        img_w, img_h = image.size
        for obj in item.get("label", []):
            cls = obj["rectanglelabels"][0]
            # Fald tilbage til faktisk billedstørrelse hvis original_* ikke findes
            ow = obj.get("original_width", img_w)
            oh = obj.get("original_height", img_h)
            x_min = obj["x"] / 100.0 * ow
            y_min = obj["y"] / 100.0 * oh
            w = obj["width"] / 100.0 * ow
            h = obj["height"] / 100.0 * oh
            x_max, y_max = x_min + w, y_min + h
            boxes.append([x_min, y_min, x_max, y_max])
            labels.append(CLASS_TO_ID[cls])

        # 3) Resize billede + skaler bokse tilsvarende
        orig_w, orig_h = image.size
        new_w, new_h = self.resize
        image = F.resize(image, [new_h, new_w])  # F.resize forventer [H, W]
        scale_x, scale_y = new_w / orig_w, new_h / orig_h

        if len(boxes) > 0:
            import torch
            boxes = torch.tensor(boxes, dtype=torch.float32)
            boxes = boxes * torch.tensor([scale_x, scale_y, scale_x, scale_y])
        else:
            boxes = torch.zeros((0, 4), dtype=torch.float32)

        labels = torch.tensor(labels, dtype=torch.int64) if len(labels) > 0 else torch.zeros((0,), dtype=torch.int64)

        # 4) Til tensor (CHW, [0,1])
        image = F.to_tensor(image)

        target = {
            "boxes": boxes,      # float[N,4] i xyxy
            "labels": labels,    # int64[N]   i [1..C], 0 er background (bruges internt)
        }
        return image, target

# 5) Collate til detection (variable antal bokse pr. billede)
def collate_fn(batch):
    return tuple(zip(*batch))

# 6) Model: Faster R-CNN ResNet-50 + FPN, med COCO-vægte og udskiftet head
def build_model(num_classes: int):
    weights = FasterRCNN_ResNet50_FPN_V2_Weights.COCO_V1  # pre-trained backbone+fpn
    model = fasterrcnn_resnet50_fpn_v2(weights=weights)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    return model

def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[info] device: {device}")

    data_root = Path(args.data_root)
    train_ds = RuneOresDataset(data_root / "train", "train", train=True,  resize=(args.img_size, args.img_size))
    val_ds   = RuneOresDataset(data_root / "val",   "val",   train=False, resize=(args.img_size, args.img_size))

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.num_workers, collate_fn=collate_fn)
    val_loader   = DataLoader(val_ds,   batch_size=args.batch_size, shuffle=False,
                              num_workers=args.num_workers, collate_fn=collate_fn)

    model = build_model(num_classes=len(CLASSES) + 1).to(device)
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(params, lr=args.lr, weight_decay=1e-4)

    args.output.mkdir(parents=True, exist_ok=True)

    for epoch in range(args.epochs):
        model.train()
        epoch_loss = 0.0
        t0 = time.time()

        for i, (images, targets) in enumerate(train_loader, 1):
            images  = [img.to(device) for img in images]
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

            optimizer.zero_grad()
            loss_dict = model(images, targets)        # dict med del-losses
            loss = sum(loss_dict.values())            # samlet loss
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

            # Enkel fremskridt hver 10. step (kan sættes lavere/højere)
            if i % 10 == 0:
                print(f"  step {i}/{len(train_loader)} loss={loss.item():.4f}")

        avg_train = epoch_loss / max(1, len(train_loader))

        # Minimal val-loss (samme loss-kald uden backprop) – vil du være HELT minimal, kan du kommentere dette ud
        val_loss = evaluate(model, val_loader, device)

        dt = time.time() - t0
        print(f"Epoch {epoch+1}/{args.epochs} | train {avg_train:.4f} | val {val_loss:.4f} | {dt:.1f}s")

        # Gem én checkpoint pr. epoch (enkelt)
        torch.save({"model": model.state_dict(), "epoch": epoch+1}, args.output / "fasterrcnn_resnet50_last.pth")

    print("[info] done.")

@torch.no_grad()
def evaluate(model, loader, device):
    was_training = model.training
    # torchvision detection forventer .train() når man vil have loss ud (quirk)
    model.train()
    total = 0.0
    n = 0
    for images, targets in loader:
        images  = [img.to(device) for img in images]
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
        loss_dict = model(images, targets)
        total += float(sum(loss_dict.values()).item())
        n += 1
    model.train(was_training)
    return total / max(1, n)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, required=True, help="Path til data_splits_ready folder")
    parser.add_argument("--output", type=Path, default=Path("./checkpoints"))
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--num-workers", type=int, default=0)  # 0 for enkel/stabil start
    parser.add_argument("--img-size", type=int, default=640)   # fast resize (W=H)
    args = parser.parse_args()
    train(args)
