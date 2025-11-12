
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
        # Attributes
        pred_x, pred_y, pred_w, pred_h, pred_c = get_bb_attributes(pred)
        gt_x, gt_y, gt_w, gt_h, gt_c = get_bb_attributes(gt)
        # Classes
        pred_class_probs = get_bb_classes(pred) 
        gt_class = get_bb_classes(gt)
        # Masks
        obj_ij = gt_c > 0 # 1 if object
        obj_ij=obj_ij.float()
        obj_i = (obj_ij.sum(dim=-1) > 0).float()[..., None]


        noobj_ij = gt_c == 0 # 1 of no object
        noobj_ij = noobj_ij.float() 
        
        epsilon = 1e-6



        # Coordinate loss
        loss_xy = self.lambda_coord * (obj_ij*((pred_x - gt_x) ** 2 + (pred_y - gt_y) ** 2)).sum()
        loss_wh = self.lambda_coord * (obj_ij * ((
            torch.sqrt(pred_w + epsilon) - 
            torch.sqrt(gt_w + epsilon))**2 +
            (torch.sqrt(pred_h + epsilon) - 
            torch.sqrt(gt_h + epsilon))**2)).sum()
        
        coord_loss = loss_xy+loss_wh
        # Confidence Loss
        
        conf_obj = (obj_ij * (pred_c-gt_c)**2).sum()
        conf_noobj = self.lambda_noobj * (noobj_ij * ((pred_c-gt_c)**2)).sum() 
        conf_loss = conf_obj+conf_noobj

        # Classification Loss
        class_loss = (obj_i * (pred_class_probs-gt_class)**2).sum() # Usikker på om den håndtere klasserne korrekt


        loss = coord_loss + conf_loss + class_loss # har ikke brugt IOU endnu??

        return loss / config.BATCH_SIZE 



def get_bb_attributes(data):
    data_attributes = data[..., :config.B*5]
    x = data_attributes[..., 0::5]
    y = data_attributes[..., 1::5]
    w = data_attributes[..., 2::5]
    h = data_attributes[..., 3::5]
    c = data_attributes[..., 4::5]
    return x, y, w, h, c

def get_bb_classes(data):
    class_probs = data[..., config.B*5:]
    return class_probs