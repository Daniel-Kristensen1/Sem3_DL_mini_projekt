import torch
from torch.utils.data import DataLoader
import torch.optim as optim
from pathlib import Path
import json


import config
from data_handler import DataHandler
from modelV2 import create_object_detector

# Sortering funktion
# collate_fn er en funktion, som bestemmer, hvordan DataLoader skal samle individuelle samples fra vores dataset til en samlet batch. 
# Normalt i PyTorch til fx klassifikations opgaver så laver DataLoader automatisk batches således her:
# images:  tensor of shape [batch_size, C, H, W]
# labels:  tensor of shape [batch_size]

# Men da vi bruger Faster R-CNN, så skal vi have output:
# {
#    "images": [img1, img2, ...],
# targets = [
#    {"boxes": ..., "labels": ...},
#    {"boxes": ..., "labels": ...},]
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
    # Tomme lister til at lave nogle loss function plots
    epoch_total_losses = []
    epoch_classification_losses = []
    epoch_box_losses = []
    epoch_object_losses = []
    epoch_rpn_box_losses = []


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
    # Fase 1: Anchor generator bliver lavet
    # Fase 2: Backbone bliver lavet
    # Fase 3: Faster R-CNN bliver lavet
    model = create_object_detector(num_classes=config.NUM_CLASSES)
    model.to(device) # Vi flytter bare her opgaven fra CPU til GPU



    # Optimizer HUSK AT SE PÅ optimizere muligheder
    # Vi optimer alle parametre (vægte)
    # Inkluderer: ResNet50 backbone, RPN (Region Proposal Netword), ROI heads (klassifikation + box regression), Anchor regression head
    optimizer = optim.SGD(# Vi anvender her Stochastic gradient descent.
        #Vi tager gradienterne fra loss.backward(), opdater vægtene i modellen i modsat retning af loss. Gør det igen og igen for hver batch
        model.parameters(),
        lr=config.LEARNING_RATE,
        momentum=0.9,# 0.9 betyder: Optimizer husk tidligere retninger, gradientstøj bliver udjævnet, træningen bliver hurtigere, modellen zigzagger mindre, den glider gennem loss-landskabet
        # weight_decay standardværdi for R-CNN = 1e-4
        weight_decay=1e-4 # Anti overfitting: Forhindrer vægtene i at blive alt for store, gør modellen generaliserer bedre, Gør RPN+ROI heads mere stabile, reducerer risikoen for "blowing up gradients"
    # Alternativer til optimizer: Adam, AdamW, RMSprop (Dog bruges denne mere i RNNs)
    # R-CNN blev designet med SGD, derfor er det den anvende optimizer her
    )

    num_epochs = config.NUM_EPOCHS

    # Mappe til at lave heckponts + tracking af bedste loss
    checkpoint_dir = Path("checkpoints")
    checkpoint_dir.mkdir(exist_ok=True)
    best_loss = float("inf")
    best_epoch = -1


    # Træn igennem antallet af epochs. Altså kom igennem træningsdataen x antal gange
    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0.0 # Skal ses som en accumulator, hvor vi samler loss for hele epoch for at kunne udregne gennemsnittet til sidst.
        epoch_cls = 0.0
        epoch_box = 0.0
        epoch_obj = 0.0
        epoch_rpn_box = 0.0


        #DataLoader vælger fx 2 samples fra datasættet, kalder DataHandler.__getitem__ for hver sample.
        #Den samler dem til en batch med collate_fn funktionen
        #Giver os images og targets for den batch
        for batch_idx, (images, targets) in enumerate(train_loader):
            # Flyt data til device (GPU/CPU)
            images = [img.to(device) for img in images] #Flytter billede til GPU fra CPU
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets] #Flytter alle tensors i dict'en til device(GPUen)


            # model(images, targets og sum(loss for loss in loss_dict.values())
            # Sender images gennem backbone, (backbone.forward) -> feature maps
            # RPN laver forslag til regioner (anchors -> proposals)
            # RPO-heads klassificerer og finjusterer bokse
            # Sammenligner output med targets (ground truth)
            # Regner 4 losses
            # De 4 losses: loss_classifier, loss_box_reg, loss_objectness, loss_rpn_box_reg. Returneres som en dictionary
            loss_dict = model(images, targets) # <- vigtigste

            # Resten under bruges kun til at lave plots
            loss_classifer = loss_dict.get("loss_classifier", 0.0)
            loss_box_reg = loss_dict.get("loss_box_reg", 0.0)
            loss_objectness = loss_dict.get("loss_objectness", 0.0)
            loss_rpn_box_reg = loss_dict.get("loss_rpn_box_reg", 0.0)


            loss = loss_classifer + loss_box_reg + loss_objectness + loss_rpn_box_reg # Laver samlet loss, til backprop

            optimizer.zero_grad() # Nulstiller gamle gradients, så vi ikke akkumulerer fra tidligere batches
            loss.backward() # PyTorch går baglæns igennem hele modellen: Beregner gradienter for backbone, RPN, heads
            optimizer.step() # Tager alle gradients, opdaterer vægtene en lille smule ud fra "learning rate" og "momentum" og "weight decay" 


            # akkumuler batch losses .item funktion er så vi får tal og ikke tensor.
            epoch_loss += loss.item()
            epoch_cls += loss_classifer.item()
            epoch_box += loss_box_reg.item()
            epoch_obj += loss_objectness.item()
            epoch_rpn_box += loss_rpn_box_reg.item()

        # Vi udregner gennemsnit pr epoch
        num_batches = len(train_loader)
        avg_epoch_loss = epoch_loss / num_batches
        avg_cls = epoch_cls / num_batches
        avg_box = epoch_box / num_batches
        avg_obj = epoch_obj / num_batches
        avg_rpn_box = epoch_rpn_box / num_batches

        # Print i terminalen for at få en status
        print(
            f"==> Epoch [{epoch+1}/{num_epochs}] "
            f"Total: {avg_epoch_loss:.4f} | "
            f"cls: {avg_cls:.4f} | box: {avg_box:.4f} | "
            f"obj: {avg_obj:.4f} | rpn_box: {avg_rpn_box:.4f}"
        )

        # Gem losses i liste
        epoch_total_losses.append(avg_epoch_loss)
        epoch_classification_losses.append(avg_cls)
        epoch_box_losses.append(avg_box)
        epoch_object_losses.append(avg_obj)
        epoch_rpn_box_losses.append(avg_rpn_box)

        

        
        # Kontrollere om nuværende epoch er bedre end sidste
        if avg_epoch_loss < best_loss:
            old_best = best_loss
            best_loss = avg_epoch_loss
            best_epoch = epoch + 1

            # Gemme model + optimizer + data
            checkpoint_path = checkpoint_dir / "best_model.pth"
            torch.save({"epoch": best_epoch, "model_state_dict": model.state_dict(), "optimizer_state_dict": optimizer.state_dict(), "loss": best_loss,}, checkpoint_path)

            if old_best < float("inf"):
                print(f"Best model at epoch {best_epoch} with loss {best_loss:.4f} (last best was {old_best:.4f})")
                
            else:
                print(f"Model saved at epoch {best_epoch} with loss {best_loss:.4f}")


    loss_log = {"total": epoch_total_losses, 
                "loss_classifier": epoch_classification_losses,
                "loss_box_reg": epoch_box_losses,
                "loss_objectness": epoch_object_losses,
                "loss_rpn_box_reg": epoch_rpn_box_losses,
                }
    
    with open(checkpoint_dir / "loss_log.json", "w") as f:
        json.dump(loss_log, f, indent=2)

    print(f"Best epoch for this training {best_epoch} with loss: {best_loss:.4f}")



if __name__ == "__main__":
    train_loop()