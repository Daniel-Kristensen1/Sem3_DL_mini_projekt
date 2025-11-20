import json
import torch
from pathlib import Path
from torch.utils.data import Dataset
from PIL import Image
from torchvision.transforms import functional as F

import config



class DataHandler(Dataset):
    def __init__(self, json_path, images_dir, resize=config.IMAGE_SIZE, train=True):
        # Gem stierne
        self.json_path = json_path
        self.images_dir = images_dir
        
        # læs JSON-filen
        self.entries = json.loads(self.json_path.read_text(encoding="utf-8"))

        # Gem resize og train
        self.resize = resize
        self.train = train

    def __len__(self):
        return len(self.entries)
    
    def __load_image__(self, entry):
        # Find billedets filnavn fra JSON-entry
        image_name = Path(entry["image"]).name

        # Fulde sti: images / filnavn
        image_path = self.images_dir / image_name

        # Åben billedet med konvertering til RGB
        image = Image.open(image_path).convert("RGB")

        return image
    
    def __get_boundingboxes_and_labels__(self, entry, img_w, img_h):
        boundingboxes = []
        labels = []

        # "label" indeholder alle vores objekter
        for obj in entry.get("label", []):
            # Klassenavn fx Gold, Adamant eller en af de andre klasser
            cls_name = obj["rectanglelabels"][0]

            # Vi skal bruge de orginale_width/-height som står i datasættet
            original_width = obj.get("original_width", img_w)
            original_height = obj.get("original_height", img_h)

            # Vi har procenter i vores JSON fil, de skal laves om til pixels
            # Husk vores xy koordinater i vores JSON fil er top-left hjørne og ikke centrum koordinater. 
            x_som_pixel = obj["x"] / 100.0 * original_width
            y_som_pixel = obj["y"] / 100.0 * original_height
            w_som_pixel = obj["width"] / 100.0 * original_width
            h_som_pixel = obj["height"] / 100.0 * original_height

            x_min = x_som_pixel
            y_min = y_som_pixel
            x_max = x_som_pixel + w_som_pixel
            y_max = y_som_pixel + h_som_pixel

            boundingboxes.append([x_min, y_min, x_max, y_max])
            labels.append(config.CLASSES_WITH_ID[cls_name])

        return boundingboxes, labels
    

    def __resize_and_scale__(self, image, boundingboxes):
        # Orginal størrelserne
        original_width, original_height = image.size

        # De nye størrelser som ændres i config.py filen
        new_width, new_height = self.resize
        image_resized = F.resize(image, [new_height, new_width])

        # Der burde ikke være billeder uden bounding boxes Længere, men bare for at være sikker
        if len(boundingboxes) == 0:
            return image_resized, []
        
        # skaler vores bounding boxes tilsvarende
        scale_x = new_width / original_width
        scale_y = new_height / original_height

        scaled_boxes = []
        
        for x_min, y_min, x_max, y_max in boundingboxes:
            x_min_scaled = x_min * scale_x
            x_max_scaled = x_max * scale_x
            y_min_scaled = y_min * scale_y
            y_max_scaled = y_max * scale_y
            scaled_boxes.append([x_min_scaled, y_min_scaled, x_max_scaled, y_max_scaled])

        return image_resized, scaled_boxes
            
    def __make_tensor__(self, image, boundingboxes, labels):
        # Laver billedet om til en tensor [C, H, W] og skalerer værdierne til [0, 1]
        image_tensor = F.to_tensor(image)

        # Laver boundingboxes om til tensors (Eller 0, hvis ingen bounding boxes er)
        if len(boundingboxes) > 0:
            boxes_tensor = torch.tensor(boundingboxes, dtype=torch.float32)
        else:
            boxes_tensor = torch.zeros((0, 4), dtype=torch.float32)
        
        # Laver labels om til tensor [N] (Eller [0] hvis ingen labels)
        if len(labels) > 0:
            labels_tensor = torch.tensor(labels, dtype=torch.int64)
        else:
            labels_tensor = torch.zeros((0,), dtype=torch.int64)
        
        target = {
            "boxes": boxes_tensor,
            "labels": labels_tensor,
        }

        return image_tensor, target
    
    def __getitem__(self, index):
        # Henter en entry fra JSON-listen
        entry = self.entries[index]

        # Åbner billedet
        image = self.__load_image__(entry)

        # Laver bounding boxes + labels (i pixels, før resize)
        img_w, img_h = image.size
        boundingboxes, labels = self.__get_boundingboxes_and_labels__(entry, img_w, img_h)
        
        # Resize billede + resize bokse tilsvarende
        image_resized, scaled_boundingboxes = self.__resize_and_scale__(image, boundingboxes)
        
        # Lav tensors + target dict
        image_tensor, target = self.__make_tensor__(image_resized, scaled_boundingboxes, labels)

        # Returner billedet og target dict
        return image_tensor, target
