from fvcore.nn import FlopCountAnalysis
import torch

from modelV2 import create_object_detector
import config

device = "cuda" if torch.cuda.is_available() else "cpu"

model = create_object_detector(num_classes=config.NUM_CLASSES).to(device)
model.eval()

dummy_image = torch.randn(3, 640, 640, device=device)
inputs = [dummy_image]  # FasterRCNN forventer en liste

flops = FlopCountAnalysis(model, inputs)
print("FLOPs: ", flops.total())
print("≈ GFLOPs: ", flops.total() / 1e9)