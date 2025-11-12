# train.py
import math, os, time, csv, random
from pathlib import Path

import torch
from torch.utils.data import DataLoader
import torch.optim as optim
import numpy as np

import config
from data_yolov1 import LSYoloV1Dataset
from architecture import SimpleCNN
from loss import LossFunc


def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def collate(batch):
    xs, ys = zip(*batch)
    return torch.stack(xs, 0), torch.stack(ys, 0)


def maybe_unpack_loss(out):
    """Supports both `loss` and `(loss, parts)` from criterion.forward"""
    if isinstance(out, tuple) and len(out) == 2 and isinstance(out[1], dict):
        loss, parts = out
        parts = {
            "coord": parts.get("coord", torch.tensor(0.0, device=loss.device)),
            "conf_obj": parts.get("conf_obj", torch.tensor(0.0, device=loss.device)),
            "conf_noobj": parts.get("conf_noobj", torch.tensor(0.0, device=loss.device)),
            "class": parts.get("class", torch.tensor(0.0, device=loss.device)),
        }
        return loss, parts
    else:
        loss = out
        zero = torch.tensor(0.0, device=loss.device) if torch.is_tensor(loss) else 0.0
        return loss, {"coord": zero, "conf_obj": zero, "conf_noobj": zero, "class": zero}


def main():
    set_seed(config.SEED)
    device = torch.device(config.DEVICE if torch.cuda.is_available() else "cpu")

    # --- Datasets & Loaders ---
    train_ds = LSYoloV1Dataset(config.TRAIN_JSON, config.TRAIN_IMAGES, img_size=config.IMAGE_SIZE[0])
    val_ds   = LSYoloV1Dataset(config.VAL_JSON,   config.VAL_IMAGES,   img_size=config.IMAGE_SIZE[0])

    train_dl = DataLoader(train_ds, batch_size=config.BATCH_SIZE, shuffle=True,  num_workers=config.WORKERS, collate_fn=collate, pin_memory=True)
    val_dl   = DataLoader(val_ds,   batch_size=config.BATCH_SIZE, shuffle=False, num_workers=config.WORKERS, collate_fn=collate, pin_memory=True)

    # --- Model, Loss, Optimizer ---
    model = SimpleCNN(num_classes=config.C).to(device)
    criterion = LossFunc()
    optimizer = optim.Adam(model.parameters(), lr=config.LEARNING_RATE, weight_decay=config.WEIGHT_DECAY)
    scaler = torch.cuda.amp.GradScaler(enabled=(device.type == "cuda"))

    # --- Dirs ---
    ckpt_dir = Path("checkpoints"); ckpt_dir.mkdir(exist_ok=True)
    runs_dir = Path("runs_yolov1"); runs_dir.mkdir(exist_ok=True)
    log_path = runs_dir / "train_log.csv"

    # CSV header
    if not log_path.exists():
        with open(log_path, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["epoch","split","loss","coord","conf_obj","conf_noobj","class","seconds"])

    best_val = math.inf

    # --- Training Loop ---
    for epoch in range(1, config.EPOCHS + 1):
        # ---------- TRAIN ----------
        model.train()
        t0 = time.time()
        run_loss = run_coord = run_cobj = run_cnoobj = run_class = 0.0

        for xb, yb in train_dl:
            xb, yb = xb.to(device, non_blocking=True), yb.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=(device.type == "cuda")):
                pred = model(xb)
                out = criterion(pred, yb)
                loss, parts = maybe_unpack_loss(out)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            run_loss   += float(loss.item())
            run_coord  += float(parts["coord"].item())
            run_cobj   += float(parts["conf_obj"].item())
            run_cnoobj += float(parts["conf_noobj"].item())
            run_class  += float(parts["class"].item())

        n_train = max(1, len(train_dl))
        train_loss  = run_loss   / n_train
        train_coord = run_coord  / n_train
        train_cobj  = run_cobj   / n_train
        train_cno   = run_cnoobj / n_train
        train_cls   = run_class  / n_train

        # ---------- VALIDATION ----------
        model.eval()
        v_loss = v_coord = v_cobj = v_cno = v_cls = 0.0
        with torch.no_grad():
            with torch.cuda.amp.autocast(enabled=(device.type == "cuda")):
                for xb, yb in val_dl:
                    xb, yb = xb.to(device, non_blocking=True), yb.to(device, non_blocking=True)
                    pred = model(xb)
                    out = criterion(pred, yb)
                    loss, parts = maybe_unpack_loss(out)

                    v_loss += float(loss.item())
                    v_coord += float(parts["coord"].item())
                    v_cobj  += float(parts["conf_obj"].item())
                    v_cno   += float(parts["conf_noobj"].item())
                    v_cls   += float(parts["class"].item())

        n_val = max(1, len(val_dl))
        val_loss = v_loss / n_val
        val_coord = v_coord / n_val
        val_cobj  = v_cobj  / n_val
        val_cno   = v_cno   / n_val
        val_cls   = v_cls   / n_val

        dt = time.time() - t0

        # --- PRINT nice line ---
        print(
            f"Epoch {epoch:03d} | "
            f"train {train_loss:.4f} (xywh {train_coord:.4f} | c_obj {train_cobj:.4f} | c_noobj {train_cno:.4f} | cls {train_cls:.4f}) | "
            f"val {val_loss:.4f} (xywh {val_coord:.4f} | c_obj {val_cobj:.4f} | c_noobj {val_cno:.4f} | cls {val_cls:.4f}) | "
            f"{dt:.1f}s"
        )

        # --- CSV log ---
        with open(log_path, "a", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow([epoch, "train", f"{train_loss:.6f}", f"{train_coord:.6f}", f"{train_cobj:.6f}", f"{train_cno:.6f}", f"{train_cls:.6f}", f"{dt:.2f}"])
            w.writerow([epoch, "val",   f"{val_loss:.6f}",   f"{val_coord:.6f}",   f"{val_cobj:.6f}",   f"{val_cno:.6f}",   f"{val_cls:.6f}",   f"{dt:.2f}"])

        # --- Save best ---
        if val_loss < best_val:
            best_val = val_loss
            torch.save({
                "epoch": epoch,
                "model": model.state_dict(),
                "opt": optimizer.state_dict(),
                "best_val": best_val,
                "config": {k: getattr(config, k) for k in dir(config) if k.isupper()},
            }, ckpt_dir / "best.pt")

    # --- Final save (last) ---
    torch.save({
        "epoch": config.EPOCHS,
        "model": model.state_dict(),
        "opt": optimizer.state_dict(),
        "best_val": best_val,
        "config": {k: getattr(config, k) for k in dir(config) if k.isupper()},
    }, ckpt_dir / "last.pt")


if __name__ == "__main__":
    main()
