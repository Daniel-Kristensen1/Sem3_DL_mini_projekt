import torch
import torch.nn as nn
from PIL import Image
from torchvision.models import resnet50, ResNet50_Weights
import config
from torchvision.transforms import functional as F
from torchvision.models.detection import fasterrcnn_resnet50_fpn_v2, FasterRCNN_ResNet50_FPN_V2_Weights
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection import FasterRCNN
from torchvision.models.detection.anchor_utils import AnchorGenerator


class Backbone(nn.Module):
    def __init__(self): 
        super().__init__()

        print("- Loading resnet50 with pretrained weights...")
        self.backbone = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
        print("- Removing final layers of resnet50...")
        self.backbone = nn.Sequential(*list(self.backbone.children())[:-2])
        print("- Defining out_channels for fasterRCNN...")
        self.out_channels = 2048
    def forward(self, x):
        print(f"- Initiating backbone forward function...")
        return {"0": self.backbone(x)} # Lavet til en tensor for at det passer med torch vision. Det fungere også uden, men der kan vist opstå problemer med edge cases.



def create_object_detector(num_classes=config.NUM_CLASSES):
      print(f"- Creating Object Detector...")
      print(f"- Creating Anchor generator...")
      anchor_gen = AnchorGenerator(
        sizes=((32, 64, 128, 256, 512),),
        aspect_ratios=((0.5, 1.0, 2.0),)
        )
      print(f"- Building model based on FasterRCNN...")
      model = FasterRCNN(
           backbone=Backbone(),
           num_classes=num_classes+1, # +1 because the background also needs a class.
           rpn_anchor_generator=anchor_gen
      )
      return model



if __name__ == "__main__":
    import torch

    NUM_CLASSES = 3   # example

    print("\n=== Building detector ===")
    model = create_object_detector(num_classes=NUM_CLASSES)

    # Put model on CPU (or CUDA if you want)
    device = torch.device("cpu")
    model.to(device)

    # Create a dummy image (3, 480, 640)
    dummy_image = torch.randn(3, 480, 640).to(device)

    # Create a dummy target (1 box)
    dummy_target = {
        "boxes": torch.tensor([[100.0, 150.0, 300.0, 350.0]]),  # [x1, y1, x2, y2]
        "labels": torch.tensor([1])                             # class ID
    }

    print("\n=== Testing TRAIN mode (loss computation) ===")
    model.train()
    train_out = model([dummy_image], [dummy_target])
    print("Loss dict:", train_out)
    print("Total loss:", sum(loss.item() for loss in train_out.values()))

    print("\n=== Testing EVAL mode (inference) ===")
    model.eval()
    with torch.no_grad():
        pred = model([dummy_image])

    print("\nPredictions:")
    print("Boxes:", pred[0]['boxes'])
    print("Labels:", pred[0]['labels'])
    print("Scores:", pred[0]['scores'])

    print("\n=== Model test complete ===")
