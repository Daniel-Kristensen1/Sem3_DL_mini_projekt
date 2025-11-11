
import torch
import config
from torch import nn as nn
from torch.nn import functional as F


class LossFunc(nn.Module):
    def __init__(self):
        super().__init__()
        self.lambda_coord = config.COORD
        self.lambda_noobj = config.NOOBJ

    def forward(self, pred, gt):
        pred_x, pred_y, pred_w, pred_h, pred_c = get_bb_attributes(pred)
        gt_x, gt_y, gt_w, gt_h, gt_c = get_bb_attributes(gt)
        obj_mask = gt_c > 0
        epsilon = 1e-6



        # Coordinate loss
        loss_xy = self.lambda_coord * (((pred_x - gt_x) ** 2 + (pred_y - gt_y) ** 2) * obj_mask).sum()
        loss_wh = self.lambda_coord * (((
            torch.sqrt(pred_w + epsilon) - 
            torch.sqrt(gt_w + epsilon))**2 +
            (torch.sqrt(pred_h + epsilon) - 
            torch.sqrt(gt_h + epsilon))**2) * 
            obj_mask
).sum()
        # Confidence Loss

        # Classification Loss




def get_bb_attributes(data):
    data_attributes = data[..., :config.B*5]
    x = data_attributes[..., 0::5]
    y = data_attributes[..., 1::5]
    w = data_attributes[..., 2::5]
    h = data_attributes[..., 3::5]
    c = data_attributes[..., 4::5]
    return x, y, w, h, c
     