
import torch
import numpy as np



def generate_anchors(base_size=16, scales=[0.5, 1.0, 2.0], aspect_ratios=[0.5, 1.0, 2.0]):
    """
    Generate anchor boxes based on scales and aspect ratios.

    Args:
        base_size (int): The size of the base anchor.
        scales (list): Scaling factors for anchors.
        aspect_ratios (list): Aspect ratios for anchors.

    Returns:
        torch.Tensor: Generated anchors of shape [num_anchors, 4] (x_min, y_min, x_max, y_max).
    """
    anchors = []
    for scale in scales:
        for ratio in aspect_ratios:
            w = base_size * scale * (ratio ** 0.5)
            h = base_size * scale / (ratio ** 0.5)
            x_min, y_min = -w / 2, -h / 2
            x_max, y_max = w / 2, h / 2
            anchors.append([x_min, y_min, x_max, y_max])
    
    return torch.tensor(anchors, dtype=torch.float32)

def non_maximum_suppression(boxes, scores, iou_threshold=0.5):
    """
    Perform non-maximum suppression (NMS) on bounding boxes.

    Args:
        boxes (torch.Tensor): Predicted boxes, shape [num_boxes, 4].
        scores (torch.Tensor): Confidence scores, shape [num_boxes].
        iou_threshold (float): IoU threshold for NMS.

    Returns:
        torch.Tensor: Indices of the retained boxes.
    """
    indices = torch.argsort(scores, descending=True)
    keep = []
    
    while indices.numel() > 0:
        current = indices[0]
        keep.append(current)
        
        if indices.numel() == 1:
            break
        
        remaining_boxes = boxes[indices[1:]]
        iou = calculate_iou(boxes[current].unsqueeze(0), remaining_boxes)
        indices = indices[1:][iou < iou_threshold]
    
    return torch.tensor(keep, dtype=torch.long)

def calculate_iou(box1, box2):
    """
    Compute IoU between two sets of boxes.
    
    Args:
        box1 (torch.Tensor): Single box, shape [1, 4].
        box2 (torch.Tensor): Multiple boxes, shape [N, 4].
    
    Returns:
        torch.Tensor: IoU scores for each box in box2.
    """
    inter = (
        torch.min(box1[:, 2], box2[:, 2]) - torch.max(box1[:, 0], box2[:, 0])
    ).clamp(0) * (
        torch.min(box1[:, 3], box2[:, 3]) - torch.max(box1[:, 1], box2[:, 1])
    ).clamp(0)
    
    box1_area = (box1[:, 2] - box1[:, 0]) * (box1[:, 3] - box1[:, 1])
    box2_area = (box2[:, 2] - box2[:, 0]) * (box2[:, 3] - box2[:, 1])
    
    union = box1_area + box2_area - inter
    return inter / union


def compute_iou(box1, box2):
    """Compute Intersection over Union (IoU) between two boxes."""
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    inter_area = max(0, x2 - x1) * max(0, y2 - y1)
    box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union_area = box1_area + box2_area - inter_area

    return inter_area / union_area if union_area > 0 else 0

def evaluate_model(model, dataloader, iou_threshold=0.5):
    """Evaluate mAP for the object detection model."""
    model.eval()
    all_precisions = []
    all_recalls = []

    for batch in dataloader:
        images = batch['image']
        true_boxes = batch['boxes']
        true_labels = batch['labels']

        with torch.no_grad():
            predictions = model(images)
        
        for i, pred in enumerate(predictions):
            pred_boxes = pred['boxes']
            pred_scores = pred['scores']
            gt_boxes = true_boxes[i]
            
            # Calculate IoU for each prediction
            matches = []
            for pred_box in pred_boxes:
                ious = [compute_iou(pred_box, gt_box) for gt_box in gt_boxes]
                matches.append(max(ious) >= iou_threshold)
            
            tp = sum(matches)
            fp = len(matches) - tp
            fn = len(gt_boxes) - tp
            
            precision = tp / (tp + fp) if tp + fp > 0 else 0
            recall = tp / (tp + fn) if tp + fn > 0 else 0
            
            all_precisions.append(precision)
            all_recalls.append(recall)

    mAP = np.mean(all_precisions)
    print(f"Mean Average Precision (mAP): {mAP:.4f}")




################
## Vizualize ##
################

import cv2
from pathlib import Path
import json

image_dir_path = Path(r"C:\Users\Daniel K\Desktop\data_splits_ready\val\images")
image_data = Path(r"C:\Users\Daniel K\Desktop\data_splits_ready\val\val.json")

CLASS_COLORS = { 
        "Adamant":        (80, 120, 70),    # dull green
        "Clay":           (200, 170, 120),  # pale brown/beige
        "Coal":           (30, 30, 30),     # black
        "Copper":         (170,100,50),     # orange/brown 
        "Gold":           (212,175,55),     # gold
        "Iron":           (130,120,110),    # gray with tint of brown
        "Mined":          (120,120,120),    # dark grey
        "Mithril":        (110,150,200),    # pale blue
        "Motherload_ore": (150,120,60),     # golden brown
        "Removable_ore":  (140,140,140),    # light grey
        "Runeite":        (45, 75, 160),    # deep blue
        "Silver":         (200,200,210),    # light silver
        "Tin":            (170,170,150),    # grey with yellow tint
    }

def get_image_path(image_index, image_dir=image_dir_path):
    return sorted(image_dir.glob("*"))[image_index]

def get_data(data_path=image_data):
    with open(data_path) as f:
        return json.load(f)


def show_first_image(image_dir, index=0):
    first_image = sorted(image_dir.glob("*"))[index]
    print(first_image)
    w, h, c = cv2.imread(str(first_image)).shape
    print(w)
    print(h)
    print(c)
    
    first_image = cv2.imread(first_image)
    cv2.imshow("Runescape image 1", first_image)

    cv2.waitKey(0)
    cv2.destroyAllWindows()

def get_image_w_h(image_path):
    h, w, _ = cv2.imread(str(image_path)).shape
    return h, w


def file_name(image_dir, index):
    first_image = sorted(image_dir.glob("*"))[index]
    filename = first_image.name
    return filename

def get_image_data(image_index, data=get_data()):

    for i in data:
        if file_name(image_dir_path, image_index) in i["image"]:
            return i
    print("Error: couldnt find image data")
            

def draw(image, x1, y1, bb_top_right, bb_lower_left, class_name, conf=1):
        rect_thickness = 2
        cv2.rectangle( #cv2.rectangle(image, start_point, end_point, color, thickness)
            image, 
            bb_top_right, 
            bb_lower_left, 
            CLASS_COLORS[class_name],
            rect_thickness
                        )
        cv2.putText(
            image,
            f"{class_name}, {conf}",
            (x1, y1 - 5),             # Text location 
            cv2.FONT_HERSHEY_SIMPLEX,           # Font
            1,                                  # Font scale
            CLASS_COLORS[class_name], # Color
            rect_thickness                      # thickness
        )



def show_first_bb(image_index):
    image_path=get_image_path(image_index)
    h, w = get_image_w_h(image_path)
    image_data = get_image_data(image_index)
    image__label_data_first_box=image_data["label"][0]


    bw = int(image__label_data_first_box["width"] / 100*w)
    bh = int(image__label_data_first_box["height"] / 100*h)
    
    
    x1 = int(image__label_data_first_box["x"] / 100*w)
    y1 = int(image__label_data_first_box["y"] / 100*h)
    x2 = x1 + bw
    y2 = y1 + bh

    img = cv2.imread(image_path)
    draw(img, x1, y1, (x1,y1), (x2, y2), image__label_data_first_box["rectanglelabels"][0])
    
    cv2.imshow("Runescape image 1", img)

    cv2.waitKey(0)
    cv2.destroyAllWindows()



def show_all_bb(image_index):
    image_path=get_image_path(image_index)
    h, w = get_image_w_h(image_path)
    image_data = get_image_data(image_index)
    img = cv2.imread(image_path)

    for bb in image_data["label"]: # bb = Bounding Box
        bw = int(bb["width"] / 100*w)
        bh = int(bb["height"] / 100*h)
    
    
        x1 = int(bb["x"] / 100*w)
        y1 = int(bb["y"] / 100*h)
        x2 = x1 + bw
        y2 = y1 + bh

    
        draw(img, x1, y1, (x1,y1), (x2, y2), bb["rectanglelabels"][0])
    
    cv2.imshow(f"Runescape image {image_index}", img)

    cv2.waitKey(0)
    cv2.destroyAllWindows()



show_all_bb(0)


#print(get_image_data(0))







#file_name()
#print(data[0]["image"])
#print(data[0]["id"])
#print(data[0]["label"][0])


#show_first_image()