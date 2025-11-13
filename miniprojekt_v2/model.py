import torch
import torch.nn as nn
from PIL import Image
from torchvision.models import resnet50, ResNet50_Weights
import config
from torchvision.transforms import functional as F
from torchvision.models.detection import fasterrcnn_resnet50_fpn_v2, FasterRCNN_ResNet50_FPN_V2_Weights
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
        


class Backbone(nn.Module):
    def __init__(self): 
        super().__init__()
        print("- Loading weights for backbone...")
        weights = FasterRCNN_ResNet50_FPN_V2_Weights.COCO_V1
        print("- Loading fasterrcnn_resnet50_fpn_v2...")
        self.backbone = fasterrcnn_resnet50_fpn_v2(weights=weights)
        print("- Removing final layers...")
        self.backbone = nn.Sequential(*list(self.backbone.children())[:-2])
    def forward(self, x):
        print(f"- Initiating backbone forward function...")
        return self.backbone(x)


class Head(nn.Module):
    def __init__(self, in_features=2048, num_classes=config.NUM_CLASSES): 
        super().__init__()
        print(f"- Creating new custom head...")
        print(f"- Defining pooling layer...")
        self.avpool = nn.AdaptiveAvgPool2d((1,1)) # hvilke variabler skal vi bruge og skal det være maxpool? Så det matcher yolo?.
        print(f"- Defining fc layer...")
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_features, 4096),
            # nn.dropout?
            nn.LeakyReLU(0.1, inplace=True),
            nn.Linear(4096, num_classes)
        )
        print( f"The custom head has been initialized with in_features:{in_features} and num_classes: {num_classes}")
    
    def forward(self, x):
        print(f"- Initiating head forward function...")
        x = self.avpool(x)
        x = self.fc(x)
        return x 

class CustomModel(nn.Module):
    def __init__(self): 
        super().__init__()
        print(f"- Creating the custom model...")
        print(f"- Defining backbone and head...")
        self.backbone = Backbone()
        self.head = Head()
    
    def forward(self, x):
        print(f"- Initiating head forward function. Combining backbone and head...")
        x = self.backbone(x)
        x = self.head(x)
        return x
