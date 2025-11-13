import torch
import torch.nn as nn
from PIL import Image
from torchvision.models import resnet50, ResNet50_Weights
import config
        
def defineBackbone():
    try:
        print("- Loading ResNet50 backbone...")
        backbone = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
        print("- Removing final layers...")
        backbone = nn.Sequential(*list(backbone.children())[:-2])
        return backbone
    except Exception as e:
        print(f"Backbone Setup... ERROR Message: {str(e)} ") 
        

class Head(nn.Module):
    def __init__(self, in_features=2048, num_classes = config.NUM_CLASSES): 
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

class CustomResModel(nn.Module):
    def __init__(self): 
        super().__init__()
        print(f"- Creating the custom model...")
        print(f"- Defining backbone and head...")
        self.backbone = defineBackbone()
        self.head = Head()
    
    def forward(self, x):
        print(f"- Initiating head forward function. Combining backbone and head...")
        x = self.backbone(x)
        x = self.head(x)
        return x
