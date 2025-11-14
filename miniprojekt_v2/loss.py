import torch.nn as nn


def loss_func(class_logits, bb_pred, targets):

    class_loss = nn.CrossEntropyLoss()
    class_loss = class_loss(class_logits, targets["label"])


    bb_loss = nn.SmoothL1Loss()
    bb_loss = bb_loss(bb_pred, targets["boxes"])

    loss = class_loss + bb_loss

    return loss


def forwad_pass(model, images, targets=None, training=False):
    class_logits, bb_pred = model(images)

    if training:
        # Training
        loss = loss_func(class_logits, bb_pred, targets)
        return loss
    else:
        # Inference
        return class_logits, bb_pred