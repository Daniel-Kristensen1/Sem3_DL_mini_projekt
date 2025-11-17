import torch
from torch.utils.data import DataLoader
import torch.optim as optim

import config
from data_handler import DataHandler
from modelV2 import create_object_detector

# Sorterings funktion
# collate_fn er en funktion, som bestemmer, hvordan DataLoader skal samle individuelle samples fra vores dataset til en samlet batch. 
# Normalt i PyTorch til fx klassifikations opgaver så laver DataLoader automatisk batches således her:
# images:  tensor of shape [batch_size, C, H, W]
# labels:  tensor of shape [batch_size]

# Men da vi bruger Faster R-CNN, så skal vi have output:
# {
#    "images": tensor of shape [batch_size, C, H, W],
#    "boxes": Tensor of shape [N, 4], 
#    "labels": Tensor of shape [N]
#}   N kan være forskelligt for hver batch altså antallet af boxes og labels


def collate_fn(batch):
    """
    Skal sørge for at vores data kommer i batches,
    som følgende:
    images = [img1, img2, ..., imgN]
    targets = [target1, target2, ..., targetN]
    """

    # batch er en liste af (img, targets) tuples. zip(*batch) betyder: 
    # Tag første element fra hver tuple -> lav liste af images
    # Tag andet element fra hver tuple -> lav liste af targets
    # Resultat: 
    # images = (image1, image2)
    # targets = (target1, target2)
    images, targets = zip(*batch)

    # Vi konverterer tuple til lister
    return list(images), list(targets)
    # Så vi får:
    # images = [image1, image2]
    # targets = [target1, target2]
    # Det er denne format R-CNN forventer at få data i



def train_loop():
    device = config.DEVICE # Vi henter vores cuda/cpu device fra config filen
    print(f"Device used for training: {device}") # Vi printer for at se hvad device der er i brug

    # Forberedelse af datasættet, som kan køres gennem DataLoader
    train_dataset = DataHandler(
        json_path = config.TRAIN_JSON,
        images_dir=config.TRAIN_IMAGES,
        resize=config.IMAGE_SIZE,
        train=True
    )

    # DataLoader får her adgang til datasættet
    train_loader = DataLoader(
        train_dataset, # Indeholder JSON, Stier, resize, bounding box processing, getitem (hent 1 sample), len (Hvor mange samples findes der)
        batch_size=config.BATCH_SIZE, # Hvor mange samples per batch. R-CNN er tung, og kræver meget, så 1-2 er ofte passende større batches kan gøre træning langsom eller umulig
        shuffle=True, # Shuffler data, altså rækkefølgen af samples hver epoch
        num_workers=config.NUM_WORKERS, # Hvor mange subprocesses der skal bruges til at loade data. 0 betyder main processen
        collate_fn=collate_fn # Vi fortæller her DataLoader at bruge vores custom collate_fn funktion til at samle batches korrekt
    )
# DataLoaderens arbejde er generelt at styre, rækkefølge, batching, shuffling, parallel loading, samling af samples, interaktionen mellem dataset og træningsloop



    # Modellen - Vi bygger Faster R-CNN modellen, inklusiv backbone, RPN, ROI-heads og klassification - regression lag
    model = create_object_detector(num_classes=config.NUM_CLASSES)
    model.to(device) # Vi flytter bare her opgaven fra CPU til GPU



    # Optimizer HUSK AT SE PÅ optimizere muligheder
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

        for batch_idx, (images, targets) in enumerate(train_loader):
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

            if (batch_idx +1) % 10 == 0:
                print(f"Epoch [{epoch+1}/{num_epochs}], Batch [{batch_idx+1}/{len(train_loader)}], Loss: {loss.item():.4f}")

        
        avg_epoch_loss = epoch_loss / len(train_loader)
        print(f"==> Epoch [{epoch+1}/{num_epochs}] "
              f"Average loss: {avg_epoch_loss:.4f}")


if __name__ == "__main__":
    train_loop()