# tools/test_loader_win.py
import torch
from torch.utils.data import DataLoader
import config
from data_yolov1 import LSYoloV1Dataset

def collate(batch):
    xs, ys = zip(*batch)
    return torch.stack(xs, 0), torch.stack(ys, 0)

def main():
    ds = LSYoloV1Dataset(config.TRAIN_JSON, config.IMAGES_DIR, img_size=config.IMAGE_SIZE[0])
    dl = DataLoader(
        ds,
        batch_size=8,
        shuffle=True,
        num_workers=2,      # kan sættes til 0 hvis du stadig vil undgå multiprocessing
        pin_memory=True,    # ok selv med CPU; harmless
        collate_fn=collate
    )

    x, y = next(iter(dl))
    print("x shape:", x.shape)  # [B,3,448,448]
    print("y shape:", y.shape)  # [B,7,7,23]
    # tæl conf-slots for B=2: indices 4 og 9
    print("antal obj i batch:", y[..., [4, 9]].sum().item())

if __name__ == "__main__":
    # På Windows: sørg for at main er guardet
    main()
