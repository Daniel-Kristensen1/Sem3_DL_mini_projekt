import json
from pathlib import Path
import matplotlib.pyplot as plt

def gotta_make_em_plots():
    log_path = Path(r"C:\Users\alext\Desktop\mappe_weights_and_loss\kmeans_anchor_change\loss_log.json")
    if not log_path.exists():
        print(f"Kan ikke finde {log_path}.")
        return

    with open(log_path, "r") as f:
        loss_log = json.load(f)

    epochs = range(1, len(loss_log["total"]) + 1)

    # 1) Total lossen
    plt.figure()
    plt.plot(epochs, loss_log["total"])
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Total training loss")
    plt.grid(True)
    plt.tight_layout()

    # 2) loss_classifieren
    plt.figure()
    plt.plot(epochs, loss_log["loss_classifier"])
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Classification loss")
    plt.grid(True)
    plt.tight_layout()

    # 3) loss_box_reg
    plt.figure()
    plt.plot(epochs, loss_log["loss_box_reg"])
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Box regression loss")
    plt.grid(True)
    plt.tight_layout()

    # 4) loss_objectness
    plt.figure()
    plt.plot(epochs, loss_log["loss_objectness"])
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Objectness loss")
    plt.grid(True)
    plt.tight_layout()

    # 5) loss_rpn_box_reg
    plt.figure()
    plt.plot(epochs, loss_log["loss_rpn_box_reg"])
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("RPN box regression loss")
    plt.grid(True)
    plt.tight_layout()

    plt.show()

if __name__ == "__main__":
    gotta_make_em_plots()
