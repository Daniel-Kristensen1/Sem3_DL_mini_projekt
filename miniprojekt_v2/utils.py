
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
