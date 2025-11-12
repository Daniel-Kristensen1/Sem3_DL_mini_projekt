import argparse
import json
import random
from pathlib import Path
import time

import numpy as np
import torch
from torch.utils.data import DataLoader

import config
from architecture import SimpleCNN
from data_yolov1 import LSYoloV1Dataset
from loss import LossFunc


def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def make_dataloaders(batch_size: int, workers: int):
    train_ds = LSYoloV1Dataset(config.TRAIN_JSON, config.TRAIN_IMAGES, img_size=config.IMAGE_SIZE[0])
    val_ds = LSYoloV1Dataset(config.VAL_JSON, config.VAL_IMAGES, img_size=config.IMAGE_SIZE[0])

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=workers, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=workers, pin_memory=True)

    return train_loader, val_loader


def save_checkpoint(path: Path, epoch: int, model: torch.nn.Module, optimizer: torch.optim.Optimizer, val_loss: float):
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "val_loss": val_loss,
        "config": {
            "S": config.S,
            "B": config.B,
            "C": config.C,
            "IMAGE_SIZE": config.IMAGE_SIZE,
        }
    }, path)


def train_one_epoch(model, loader, loss_fn, optimizer, device):
    model.train()
    total_loss = 0.0
    total_samples = 0
    accum_parts = {"coord": 0.0, "conf_obj": 0.0, "conf_noobj": 0.0, "class": 0.0}

    for imgs, targets in loader:
        bs = imgs.size(0)
        imgs = imgs.to(device, dtype=torch.float32)
        targets = targets.to(device, dtype=torch.float32)

        preds = model(imgs)
        loss, parts = loss_fn(preds, targets)

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        optimizer.step()

        # Loss and parts are currently sums (per the LossFunc implementation)
        total_loss += loss.item()
        total_samples += bs

        for k in accum_parts:
            accum_parts[k] += parts[k].item()

    avg_loss = total_loss / max(1, total_samples)
    for k in accum_parts:
        accum_parts[k] = accum_parts[k] / max(1, total_samples)

    return avg_loss, accum_parts


def evaluate(model, loader, loss_fn, device):
    model.eval()
    total_loss = 0.0
    total_samples = 0
    accum_parts = {"coord": 0.0, "conf_obj": 0.0, "conf_noobj": 0.0, "class": 0.0}

    with torch.no_grad():
        for imgs, targets in loader:
            bs = imgs.size(0)
            imgs = imgs.to(device, dtype=torch.float32)
            targets = targets.to(device, dtype=torch.float32)

            preds = model(imgs)
            loss, parts = loss_fn(preds, targets)

            total_loss += loss.item()
            total_samples += bs

            for k in accum_parts:
                accum_parts[k] += parts[k].item()

    avg_loss = total_loss / max(1, total_samples)
    for k in accum_parts:
        accum_parts[k] = accum_parts[k] / max(1, total_samples)

    return avg_loss, accum_parts


def run_training(epochs: int, lr: float, batch_size: int, workers: int, device_str: str, model_dir: str):
    device = torch.device(device_str if torch.cuda.is_available() and "cuda" in device_str else "cpu")
    set_seed(config.SEED)

    train_loader, val_loader = make_dataloaders(batch_size, workers)

    model = SimpleCNN(num_classes=config.C, in_channels=3)
    model = model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=config.WEIGHT_DECAY)
    loss_fn = LossFunc()

    best_val_loss = float("inf")
    model_dir = Path(model_dir)
    start_time = time.time()

    for epoch in range(1, epochs + 1):
        t0 = time.time()
        train_loss, train_parts = train_one_epoch(model, train_loader, loss_fn, optimizer, device)
        val_loss, val_parts = evaluate(model, val_loader, loss_fn, device)
        epoch_time = time.time() - t0

        print(f"Epoch {epoch:03d}/{epochs} - time: {epoch_time:.1f}s")
        print(f"  Train loss: {train_loss:.6f} | coord: {train_parts['coord']:.6f} conf_obj: {train_parts['conf_obj']:.6f} conf_noobj: {train_parts['conf_noobj']:.6f} class: {train_parts['class']:.6f}")
        print(f"  Val   loss: {val_loss:.6f} | coord: {val_parts['coord']:.6f} conf_obj: {val_parts['conf_obj']:.6f} conf_noobj: {val_parts['conf_noobj']:.6f} class: {val_parts['class']:.6f}")

        # Save best
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            save_checkpoint(model_dir / "best_model.pth", epoch, model, optimizer, val_loss)
            print(f"  Saved new best model (val_loss={val_loss:.6f})")

        # Save last for every epoch (keeps most recent)
        save_checkpoint(model_dir / "last_model.pth", epoch, model, optimizer, val_loss)

    total_time = time.time() - start_time
    print(f"Training completed in {total_time/60:.2f} minutes. Best val loss: {best_val_loss:.6f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train YOLOv1-style model")
    parser.add_argument("--epochs", type=int, default=config.EPOCHS)
    parser.add_argument("--lr", type=float, default=config.LEARNING_RATE)
    parser.add_argument("--batch-size", type=int, default=config.BATCH_SIZE)
    parser.add_argument("--workers", type=int, default=config.WORKERS)
    parser.add_argument("--device", type=str, default=config.DEVICE)
    parser.add_argument("--model-dir", type=str, default="./models")
    args = parser.parse_args()

    run_training(args.epochs, args.lr, args.batch_size, args.workers, args.device, args.model_dir)