import torch
import torch.nn as nn
from PIL import Image
from torchvision.models import resnet50, ResNet50_Weights
        

# Backbone ResNet
backbone = resnet50(weights=ResNet50_Weights.DEFAULT)
backbone.requires_grad_(False)

print("begin printing...")
print(backbone)