# Antal parametre i modellen 
from model import CustomModel
import config
import torch

device = config.DEVICE  # fx "cuda" eller "cpu"

model = CustomModel().to(device)

# Totalt antal parametre
total_params = sum(p.numel() for p in model.parameters())
# Kun trænbare parametre
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

print(f"Total parameters: {total_params:,}")
print(f"Trainable parameters: {trainable_params:,}")




# FLOPS (operations)
#pip install thop
from thop import profile
from model import CustomModel
import config
import torch

device = "cpu"  # til FLOP-analysen er CPU fint

model = CustomModel().to(device)
model.eval()

# Dummy input med samme størrelse som jeres billeder
B = 1
C = 3
H, W = config.IMAGE_SIZE
dummy_input = torch.randn(B, C, H, W).to(device)

# profile returnerer FLOPs og params
flops, params = profile(model, inputs=(dummy_input,))

print(f"FLOPs: {flops:,}")
print(f"Params (from thop): {params:,}")



# Gemme trænings- og val-loss og lave plots

# Vi har 2 lister i train.py:
train_losses = []
val_losses = []

# Inde i epoch-loopet kan vi implementere følgende:
for epoch in range(1, num_epochs + 1):
    train_loss = train_one_epoch(...)
    val_loss = eval_one_epoch(...)

    train_losses.append(train_loss)
    val_losses.append(val_loss)

# Efter træningen (Stadig i train.py)
import matplotlib.pyplot as plt

plt.figure()
plt.plot(range(1, len(train_losses) + 1), train_losses, label="Train loss")
plt.plot(range(1, len(val_losses) + 1), val_losses, label="Val loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Training and validation loss")
plt.legend()
plt.grid(True)
plt.savefig("loss_curve.png", dpi=200)
plt.close()



# Tabeller med kvantitative resultater
#Hvis vi fx måler noget simpelt som:
#gennemsnitlig loss på test-sæt
#måske mAP, IoU, eller bare “hit rate”
#kan vi logge det i en liste af dicts og gemme som CSV / Markdown.
import pandas as pd

results = []

# eksempel: efter træning
results.append({
    "model": "CustomDetector_v1",
    "epochs": num_epochs,
    "train_loss": float(train_losses[-1]),
    "val_loss": float(val_losses[-1]),
    "gflops": float(gflops),
    "params_million": total_params / 1e6,
})

df = pd.DataFrame(results)
df.to_csv("results.csv", index=False)

print(df.to_markdown(index=False))


# Hvis vi vil have kvalitative resultater (billeder med bokse)
#Her er et simpelt eksempel på at:
#Læse et billede fra test-sættet
#Køre modellen
#Tegne de top-k bokse
#Gemme billedet på disk
#Hvis vi bruger torchvision ops, er der en helper til at tegne bokse:
import torch
from torchvision.utils import draw_bounding_boxes
from torchvision.transforms.functional import to_pil_image
from model import CustomModel
import config
from data_handler import DataHandler

device = config.DEVICE

# Load model + weights
model = CustomModel().to(device)
ckpt = torch.load("checkpoints/custom_detector_best.pth", map_location=device)
model.load_state_dict(ckpt["model"])
model.eval()

# Lav et test-dataset
test_dataset = DataHandler(
    json_path=config.TEST_JSON,
    images_dir=config.TEST_IMAGES,
    resize=config.IMAGE_SIZE,
    train=False,
)

# Tag et sample
image, target = test_dataset[0]  # image: [3,H,W], target: {"boxes": [N,4], "labels": [N]}
image_batch = image.unsqueeze(0).to(device)  # [1,3,H,W]

with torch.no_grad():
    class_logits, box_preds = model(image_batch)
    # Her afhænger det 100% af jeres output-format + NMS-logik
    # For eksempel kunne I have en funktion:
    #   boxes, scores, labels = postprocess(class_logits, box_preds, score_thresh=0.5)
    # som giver:
    #   boxes: [K,4], labels: [K]

# For demo, lad os sige I nu har:
boxes = target["boxes"]          # brug true bokse til qualitative (til at starte med)
labels = target["labels"]

# Tegn dem
img_uint8 = (image * 255).to(torch.uint8)  # [3,H,W]
boxes_to_draw = boxes  # [N,4]

img_with_boxes = draw_bounding_boxes(
    img_uint8,
    boxes=boxes_to_draw,
    labels=[str(int(l.item())) for l in labels],
    width=2,
)

pil_img = to_pil_image(img_with_boxes)
pil_img.save("qualitative_example_0_gt.png")
