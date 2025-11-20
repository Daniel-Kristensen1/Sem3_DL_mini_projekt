import torch.nn as nn
from torchvision.models import resnet50, ResNet50_Weights
from torchvision.models.detection import FasterRCNN

import config



class Backbone(nn.Module):
    def __init__(self): 
        super().__init__()
        print("- Initiating Backbone...")
        print("- Loading resnet50 with pretrained weights...")
        self.backbone = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
        print("- Removing final layers of resnet50...")
        self.backbone = nn.Sequential(*list(self.backbone.children())[:-2])
        print("- Defining out_channels for fasterRCNN...")
        self.out_channels = 2048 # FasterRCNN forventer en output_channels som har en værdi på 2048
    def forward(self, x):
        #(Kun brugt til eventuel debug) -> print(f"- Initiating backbone forward function...")
        return {"0": self.backbone(x)} # Lavet til en tensor for at det passer med torch vision. Det fungere også uden, men der kan vist opstå problemer med edge cases.



def create_object_detector(num_classes=config.NUM_CLASSES):
      print(f"- Creating Object Detector...")
      print(f"- Creating Anchor generator...")
      anchor_gen = config.K_MEAN_OPTIMIZED_ANCHORS
      print(f"- Building model based on FasterRCNN...")
      model = FasterRCNN(
           backbone=Backbone(),
           num_classes=num_classes+1, # +1 because the background also needs a class.
           rpn_anchor_generator=anchor_gen
      )
      return model