# tools/test_loader.py
from torch.utils.data import DataLoader
import torch
import config
from data_yolov1 import LSYoloV1Dataset

ds = LSYoloV1Dataset(config.TRAIN_JSON, config.IMAGES_DIR, img_size=config.IMAGE_SIZE[0])

# Collate-funktion der stabler batch korrekt
def collate(batch):
    xs, ys = zip(*batch)
    return torch.stack(xs, 0), torch.stack(ys, 0)

dl = DataLoader(ds, batch_size=8, shuffle=True, num_workers=2, collate_fn=collate)

x, y = next(iter(dl))
print("x shape:", x.shape)  # forventes [8,3,448,448]
print("y shape:", y.shape)  # forventes [8,7,7,23]
print("Antal objekter (conf=1) i batch:", y[..., [4,9]].sum().item())  # tæl boks-slots
