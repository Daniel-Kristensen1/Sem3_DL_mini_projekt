import torch
import torch.optim as optim
import config
import data_handler

def training_the_model(model, datahandler, optimizer, device):

    for batch, (images, targets) in enumerate(datahandler):
        images = [img.to(device) for img in images]
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]