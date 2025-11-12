import pathlib
import torch
from PIL import Image, ImageDraw, ImageFont
from torchvision.transforms import functional as F

from train_detector import build_model, CLASSES

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = build_model(num_classes=len(CLASSES) + 1)
try:
    ckpt = torch.load(
        "checkpoints/fasterrcnn_resnet50_best.pth",
        map_location=device,
        weights_only=True,
    )
except TypeError:
    ckpt = torch.load(
        "checkpoints/fasterrcnn_resnet50_best.pth",
        map_location=device,
    )
state_dict = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt
model.load_state_dict(state_dict)
model.to(device).eval()

image_path = pathlib.Path(
    r"C:\Program Files (x86)\Programmering\3_semester\Mini_projekt2.0\Billed til test\cap_20251022_161211_571993_bde478.png"
)
image = Image.open(image_path).convert("RGB")
tensor = F.to_tensor(image).to(device)

with torch.no_grad():
    outputs = model([tensor])[0]

draw_image = image.copy()
drawer = ImageDraw.Draw(draw_image)
font = ImageFont.load_default()

score_threshold = 0.5
palette = [
    "#FF6B6B",
    "#4ECDC4",
    "#C7F464",
    "#FFB347",
    "#A8A5E6",
    "#FF9DE2",
    "#6BCB77",
    "#4D96FF",
    "#FFD93D",
    "#843b62",
    "#3E517A",
    "#F95738",
    "#66C3FF",
]

for idx, (box, label, score) in enumerate(
    zip(outputs["boxes"], outputs["labels"], outputs["scores"])
):
    if score < score_threshold:
        continue
    color = palette[idx % len(palette)]
    cls_name = CLASSES[label.item() - 1]
    x1, y1, x2, y2 = box.tolist()
    drawer.rectangle([x1, y1, x2, y2], outline=color, width=3)
    caption = f"{cls_name} {score:.2f}"
    text_size = drawer.textbbox((x1, y1), caption, font=font)
    drawer.rectangle(
        [x1, y1 - (text_size[3] - text_size[1]) - 4, x1 + (text_size[2] - text_size[0]) + 4, y1],
        fill=color,
    )
    drawer.text((x1 + 2, y1 - (text_size[3] - text_size[1]) - 2), caption, fill="black", font=font)

output_path = image_path.with_name(image_path.stem + "_pred.png")
draw_image.save(output_path)
print(f"Saved visualization to {output_path}")
