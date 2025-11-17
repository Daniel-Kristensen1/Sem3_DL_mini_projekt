import torch
from torch.utils.data import DataLoader
import torch.optim as optim

import config
from data_handler import DataHandler
from modelV2 import create_object_detector

# Sorterings funktion
def sorter(batch):
    """
    Skal sørge for at vores data kommer i batches,
    som følgende:
    images = [img1, img2, ..., imgN]
    targets = [target1, target2, ..., targetN]
    """

    images, targets = zip(*batch)
    return list(images), list(targets)


def train_loop():
    device = config.DEVICE
    print(f"Device used for training: {device}")

    # Dataset plus dataloader
    train_dataset = DataHandler(
        json_path = config.TRAIN_JSON,
        images_dir=config.TRAIN_IMAGES,
        resice=config.IMAGE_SIZE,
        train=True
    )


    train_loader = DataLoader(
        train_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=True,
        num_workers=config.NUM_WORKERS,
        sorter=sorter
    )

    # Modellen
    model = create_object_detector(num_classes=config.NUM_CLASSES)
    model.to(device)



    # Optimizer
    optimizer = optim.SGD(
        model.parameters(),
        lr=config.LEARNING_RATE,
        momentum=0.9,
        weight_decay=1e-4
    )

    num_epochs = config.NUM_EPOCHS

    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0.0

        for batch, (images, targets) in enumerate(train_loader):
            # Flyt data til device (GPU/CPU)
            images = [img.to(device) for img in images]
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

            # Modellen kaldes, for en dict med losses
            loss_dict = model(images, targets)
            loss = sum(loss for loss in loss_dict.values())

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

            if (batch +1) % 10 == 0:
                print(f"Epoch [{epoch+1}/{num_epochs}], Batch [{batch+1}/{len(train_loader)}], Loss: {loss.item():.4f}")

        
        avg_epoch_loss = epoch_loss / len(train_loader)
        print(f"==> Epoch [{epoch+1}/{num_epochs}] "
              f"Average loss: {avg_epoch_loss:.4f}")