# train.py
import math, time
from pathlib import Path

import torch
from torch.utils.data import DataLoader
import torch.optim as optim

import config
from data_yolov1 import LSYoloV1Dataset
from architecture import SimpleCNN
from loss import LossFunc

def collate(batch):
    xs, ys = zip(*batch)
    return torch.stack(xs, 0), torch.stack(ys, 0)

def set_seed(s=42):
    import random, numpy as np
    random.seed(s); np.random.seed(s); torch.manual_seed(s); torch.cuda.manual_seed_all(s)

def main():
    set_seed(config.SEED)
    device = torch.device(config.DEVICE if torch.cuda.is_available() else "cpu")

    # --- datasets (peg på matchende images-mapper) ---
    train_ds = LSYoloV1Dataset(config.TRAIN_JSON, config.TRAIN_IMAGES, img_size=config.IMAGE_SIZE[0])
    val_ds   = LSYoloV1Dataset(config.VAL_JSON,   config.VAL_IMAGES,   img_size=config.IMAGE_SIZE[0])

    # --- dataloaders (start med num_workers=0 på Windows) ---
    train_dl = DataLoader(train_ds, batch_size=config.BATCH_SIZE, shuffle=True,
                          num_workers=0, collate_fn=collate, drop_last=False)
    val_dl   = DataLoader(val_ds,   batch_size=config.BATCH_SIZE, shuffle=False,
                          num_workers=0, collate_fn=collate, drop_last=False)

    # --- model, loss, optimizer ---
    model = SimpleCNN().to(device)
    criterion = LossFunc()
    optimizer = optim.AdamW(model.parameters(), lr=config.LEARNING_RATE, weight_decay=config.WEIGHT_DECAY)

    scaler = torch.cuda.amp.GradScaler(enabled=(device.type == "cuda"))
    ckpt_dir = Path("checkpoints"); ckpt_dir.mkdir(exist_ok=True)
    best_val = math.inf

    for epoch in range(1, config.EPOCHS + 1):
        # ---------- train ----------
        model.train()
        t0 = time.time()
        running = 0.0
        for xb, yb in train_dl:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=(device.type == "cuda")):
                pred = model(xb)
                loss = criterion(pred, yb)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            running += loss.item()

        train_loss = running / max(1, len(train_dl))

        # ---------- val ----------
        model.eval()
        vl = 0.0
        with torch.no_grad(), torch.cuda.amp.autocast(enabled=(device.type == "cuda")):
            for xb, yb in val_dl:
                xb, yb = xb.to(device), yb.to(device)
                pred = model(xb)
                vl += criterion(pred, yb).item()
        val_loss = vl / max(1, len(val_dl))

        dt = time.time() - t0
        print(f"Epoch {epoch:03d} | train {train_loss:.4f} | val {val_loss:.4f} | {dt:.1f}s")

        # save best
        if val_loss < best_val:
            best_val = val_loss
            torch.save({"epoch": epoch,
                        "model": model.state_dict(),
                        "opt": optimizer.state_dict()},
                       ckpt_dir / "best.pt")

    # final save
    torch.save({"epoch": config.EPOCHS,
                "model": model.state_dict(),
                "opt": optimizer.state_dict()},
               ckpt_dir / "last.pt")

if __name__ == "__main__":
    main()
