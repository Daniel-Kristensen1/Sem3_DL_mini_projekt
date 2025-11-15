#!/usr/bin/env python
import argparse
from pathlib import Path

import torch
from torch.utils.data import DataLoader

import config
from data_handler import DataHandler
from model import CustomModel  # din backbone+anchor+head model


def detection_collate(batch):
    images = []
    boxes = []
    labels = []

    for img, tgt in batch:
        images.append(img)              # [3,H,W]
        boxes.append(tgt["boxes"])      # [Ni,4]
        labels.append(tgt["labels"])    # [Ni]

    images = torch.stack(images, dim=0)  # [B,3,H,W]
    return images, boxes, labels


def detection_loss(class_logits, box_preds, gt_boxes, gt_labels):
    """
    TODO: Implementér jeres rigtige detektions-loss her,
    baseret på anchor-matching osv.
    """
    loss_cls = class_logits.mean() * 0.0
    loss_box = box_preds.mean() * 0.0
    loss = loss_cls + loss_box
    return loss, loss_cls, loss_box


def train_one_epoch(model, dataloader, optimizer, device):
    model.train()
    total_loss = 0.0
    total_cls_loss = 0.0
    total_box_loss = 0.0

    for batch_idx, (images, gt_boxes, gt_labels) in enumerate(dataloader):
        images = images.to(device)
        gt_boxes = [b.to(device) for b in gt_boxes]
        gt_labels = [l.to(device) for l in gt_labels]

        class_logits, box_preds = model(images)

        loss, loss_cls, loss_box = detection_loss(
            class_logits, box_preds, gt_boxes, gt_labels
        )

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        total_cls_loss += loss_cls.item()
        total_box_loss += loss_box.item()

        if (batch_idx + 1) % 10 == 0:
            print(
                f"[batch {batch_idx+1}/{len(dataloader)}] "
                f"loss: {loss.item():.4f} "
                f"(cls: {loss_cls.item():.4f}, box: {loss_box.item():.4f})"
            )

    n = max(1, len(dataloader))
    return (
        total_loss / n,
        total_cls_loss / n,
        total_box_loss / n,
    )


@torch.no_grad()
def eval_one_epoch(model, dataloader, device):
    model.eval()
    total_loss = 0.0
    total_cls_loss = 0.0
    total_box_loss = 0.0

    for images, gt_boxes, gt_labels in dataloader:
        images = images.to(device)
        gt_boxes = [b.to(device) for b in gt_boxes]
        gt_labels = [l.to(device) for l in gt_labels]

        class_logits, box_preds = model(images)
        loss, loss_cls, loss_box = detection_loss(
            class_logits, box_preds, gt_boxes, gt_labels
        )

        total_loss += loss.item()
        total_cls_loss += loss_cls.item()
        total_box_loss += loss_box.item()

    n = max(1, len(dataloader))
    return (
        total_loss / n,
        total_cls_loss / n,
        total_box_loss / n,
    )


def parse_args():
    parser = argparse.ArgumentParser(description="Train custom multi-box detector")
    parser.add_argument("--epochs", type=int, default=config.NUM_EPOCHS)
    parser.add_argument("--batch-size", type=int, default=config.BATCH_SIZE)
    parser.add_argument("--lr", type=float, default=config.LEARNING_RATE)
    parser.add_argument("--output", type=str, default="checkpoints")
    return parser.parse_args()


def main():
    args = parse_args()
    device = config.DEVICE

    # Datasets
    train_dataset = DataHandler(
        json_path=config.TRAIN_JSON,
        images_dir=config.TRAIN_IMAGES,
        resize=config.IMAGE_SIZE,
        train=True,
    )
    val_dataset = DataHandler(
        json_path=config.VAL_JSON,
        images_dir=config.VAL_IMAGES,
        resize=config.IMAGE_SIZE,
        train=False,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=detection_collate,
        num_workers=4,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=detection_collate,
        num_workers=4,
    )

    # Model
    model = CustomModel().to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    best_val_loss = float("inf")

    for epoch in range(1, args.epochs + 1):
        print(f"\nEpoch {epoch:03d}/{args.epochs}")

        train_loss, train_cls, train_box = train_one_epoch(model, train_loader, optimizer, device)
        val_loss, val_cls, val_box = eval_one_epoch(model, val_loader, device)

        print(
            f"train_loss: {train_loss:.4f} (cls: {train_cls:.4f}, box: {train_box:.4f}) | "
            f"val_loss: {val_loss:.4f} (cls: {val_cls:.4f}, box: {val_box:.4f})"
        )

        # Gem "last"
        last_path = output_dir / "custom_detector_last.pth"
        torch.save(
            {
                "epoch": epoch,
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "train_loss": train_loss,
                "val_loss": val_loss,
            },
            last_path,
        )

        # Gem "best"
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_path = output_dir / "custom_detector_best.pth"
            torch.save(
                {
                    "epoch": epoch,
                    "model": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "train_loss": train_loss,
                    "val_loss": val_loss,
                },
                best_path,
            )
            print(f"  -> Saved new BEST model to {best_path}")


if __name__ == "__main__":
    main()

