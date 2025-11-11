from pathlib import Path
import json, cv2, numpy as np, torch 
from torch.utils.data import Dataset
import config

LABEL2ID = {n: i for i, n in enumerate (config.CLASSES)}

class LSYoloV1Dataset(Dataset):
    """
    Loader billeder + labels fra Label Studio JSON og laver targets i form [S, S, B*5 + C]
    """

    def __init__ (self, json_path, images_dir, img_size=config.IMAGE_SIZE[0]):
        self.items = json.loads(Path(json_path).read_text(encoding="utf-8"))
        self.images_dir = Path(images_dir)
        self.S, self.B, self.C = config.S, config.B, config.C
        self.img_size = img_size

    def __len__(self):
        return len(self.items)
    
    def __getitem__(self, idx):
        it = self.items[idx]
        p = self.images_dir / Path(it["image"]).name
        img = cv2.imread(str(p))
        if img is None:
            raise FileNotFoundError(p)
        
        # Konvertering til RGB farver og rescaling til config.IMAGE_SIZE
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (self.img_size, self.img_size))
        x = torch.from_numpy(img.astype(np.float32) / 255).permute(2, 0, 1)

        # YOLOv1-target [S,S,B*5+C]
        target = torch.zeros(self.S, self.S, self.B * 5 + self.C, dtype=torch.float32)

        for ann in it.get("label", []):
            cx = (ann["x"] + ann["width"] / 2) / 100.0
            cy = (ann["y"] + ann["height"] / 2) / 100.0
            w  = ann["width"]  / 100.0
            h  = ann["height"] / 100.0
            lab = ann.get("rectanglelabels", ["_NA_"])[0]
            if lab not in LABEL2ID:
                continue
            c = LABEL2ID[lab]
            i = min(int(cx * self.S), self.S - 1)
            j = min(int(cy * self.S), self.S - 1)

            # læg i første ledige boks-slot
            for b in range(self.B):
                conf_idx = b * 5 + 4
                if target[j, i, conf_idx] == 0:
                    base = b * 5
                    target[j, i, base + 0] = cx
                    target[j, i, base + 1] = cy
                    target[j, i, base + 2] = w
                    target[j, i, base + 3] = h
                    target[j, i, base + 4] = 1.0
                    target[j, i, self.B * 5 + c] = 1.0
                    break

        return x, target

        