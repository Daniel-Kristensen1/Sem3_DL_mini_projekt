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

        print("- Loading resnet50 with pretrained weights...")
        self.backbone = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
        print("- Removing final layers of resnet50...")
        self.backbone = nn.Sequential(*list(self.backbone.children())[:-2])
    def forward(self, x):
        print(f"- Initiating backbone forward function...")
        return self.backbone(x)


class Head(nn.Module): # https://medium.com/@noel.benji/customizing-object-detection-models-with-lightweight-pytorch-code-ed043e48a460
    def __init__(self, in_features=2048, num_classes=config.NUM_CLASSES+1): 
        super().__init__()
        print(f"- Creating new custom head...")
        self.avpool = nn.AdaptiveAvgPool2d((1,1)) # hvilke variabler skal vi bruge og skal det være maxpool? Så det matcher yolo?.
        print(f"- Defining fc class layer...")
        self.class_head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_features, 4096), # ######### usikkker på outchannels #########
            # nn.dropout?
            nn.LeakyReLU(0.1, inplace=True),   # ##### hav et argument for hvorfor denne aktiveringsfunktion #######3
            nn.Linear(4096, num_classes) # Output = logits for hver klasse
        )
        print(f"- Defining fc bounding box layer...")
        self.bb_head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_features, 4096), # ######### usikkker på outchannels #########
            # nn.dropout?
            nn.LeakyReLU(0.1, inplace=True),   # ##### hav et argument for hvorfor denne aktiveringsfunktion #######3
            nn.Linear(4096, 4) # Output = boundingbox koordinater: [x_min, y_min, x_max, y_max]
        )

        print( f"The custom head has been initialized with in_features:{in_features} and num_classes: {num_classes}")
    
    def forward(self, x):
        print(f"- Initiating head forward function...")
        x = self.avpool(x)
        class_logits = self.class_head(x)
        bb_pred = self.bb_head(x)
        return class_logits, bb_pred

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
        class_logits, bb_pred = self.head(x)
        return class_logits, bb_pred



#####################
### TEST AF MODEL ###
#####################

# Create an instance of the backbone
backbone = Backbone()

# Check the model summary
print(backbone)


# Example input (batch of 2 images, 3 channels, 224x224)
dummy_input = torch.randn(2, 3, 224, 224)

model = CustomModel()
# Forward pass
class_logits, bbox_predictions = model(dummy_input)

# Output sizes
print("Bounding Box Predictions:", bbox_predictions.size())  # Should be [batch_size, 4]
print("Class Predictions:", class_logits.size())  # Should be [batch_size, num_classes]