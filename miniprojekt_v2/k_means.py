#!/usr/bin/env python
import json
from pathlib import Path
from collections import Counter

import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans


# ==== KONFIGURATION ====
JSON_PATH = r"C:\Program Files (x86)\Programmering\3_semester\Mini_projekt2.0\data_splits_ready\train\train.json"   # tilpas sti hvis nødvendigt
IMG_SIZE = 640             # vi resizer til 640x640
N_CLUSTERS = 3             # antal k-means clusters (anker-størrelser)


def load_bbox_sizes(json_path, img_size=640):
    """
    Loader alle bounding boxes fra Label Studio JSON
    og returnerer deres (width_px, height_px) i et 640x640 billede.

    Antagelse: x, y, width, height er i procent (0-100) af billedets størrelse.
    Label Studio gør typisk:
        x, y, width, height i % af billedets bredde/højde.
    """
    json_path = Path(json_path)
    data = json.loads(json_path.read_text(encoding="utf-8"))

    widths_px = []
    heights_px = []

    for item in data:
        labels = item.get("label", [])
        for box in labels:
            w_pct = box.get("width", None)
            h_pct = box.get("height", None)

            # Skip hvis noget mangler eller er 0
            if w_pct is None or h_pct is None:
                continue
            if w_pct <= 0 or h_pct <= 0:
                continue

            # Konverter til pixel i et 640x640 billede:
            w_px = (w_pct / 100.0) * img_size
            h_px = (h_pct / 100.0) * img_size

            widths_px.append(w_px)
            heights_px.append(h_px)

    boxes = np.column_stack([widths_px, heights_px])
    return boxes


def run_kmeans(boxes, n_clusters=5, random_state=42):
    """
    Kører K-Means på (width, height) og returnerer:
    - model
    - cluster labels
    - cluster centre
    """
    kmeans = KMeans(
        n_clusters=n_clusters,
        n_init="auto",  # nyere sklearn
        random_state=random_state,
    )
    kmeans.fit(boxes)
    labels = kmeans.labels_
    centers = kmeans.cluster_centers_
    return kmeans, labels, centers


def print_cluster_info(centers, labels):
    """
    Printer info om hver cluster:
    - gennemsnitlig width/height (centroid)
    - hvor mange bokse der ligger i hvert cluster
    """
    counts = Counter(labels)
    print("\n=== K-Means cluster centres (i pixels, resized til 640x640) ===")
    for idx, (w, h) in enumerate(centers):
        n = counts.get(idx, 0)
        print(
            f"Cluster {idx}: "
            f"width ≈ {w:.1f}px, height ≈ {h:.1f}px, "
            f"antal bokse = {n}"
        )

    # Sortér efter areal (w*h) – ofte interessant ift. anchor sizes
    print("\n=== Cluster centres sorteret efter areal ===")
    centers_with_area = [
        (i, c[0], c[1], c[0] * c[1]) for i, c in enumerate(centers)
    ]
    centers_with_area.sort(key=lambda x: x[3])
    for idx, w, h, area in centers_with_area:
        print(
            f"Cluster {idx}: "
            f"width ≈ {w:.1f}px, height ≈ {h:.1f}px, area ≈ {area:.0f}"
        )


def plot_clusters(boxes, labels, centers, img_size=640):
    """
    Laver et lækkert scatter plot:
    - punkter = alle (width, height) i pixels
    - farve = cluster
    - X markerer centroid for hver cluster
    """
    plt.figure(figsize=(8, 8))

    # Scatter med farver efter cluster
    scatter = plt.scatter(
        boxes[:, 0],
        boxes[:, 1],
        c=labels,
        alpha=0.4,
        s=20,
    )

    # Plot cluster centres
    plt.scatter(
        centers[:, 0],
        centers[:, 1],
        marker="X",
        s=200,
        edgecolor="black",
        linewidths=1.5,
    )

    # Annotér centre med (w,h)
    for i, (w, h) in enumerate(centers):
        plt.text(
            w,
            h,
            f"{i}: {w:.0f}×{h:.0f}",
            fontsize=9,
            ha="left",
            va="bottom",
        )

    plt.title(f"K-Means på bbox-størrelser (resized til {img_size}×{img_size})")
    plt.xlabel("Width (pixels)")
    plt.ylabel("Height (pixels)")
    plt.grid(True, linestyle="--", alpha=0.3)
    plt.axis("equal")  # så skala på x/y matcher
    plt.tight_layout()
    plt.show()


def main():
    # 1) Load bounding boxes
    boxes = load_bbox_sizes(JSON_PATH, img_size=IMG_SIZE)
    print(f"Antal bokse fundet: {len(boxes)}")

    if len(boxes) == 0:
        print("Ingen bokse fundet i JSON – check fil og struktur.")
        return

    # 2) Run K-Means
    kmeans, labels, centers = run_kmeans(
        boxes,
        n_clusters=N_CLUSTERS,
        random_state=42,
    )

    # 3) Print resultater i pixels
    print_cluster_info(centers, labels)

    # 4) Plot resultatet
    plot_clusters(boxes, labels, centers, img_size=IMG_SIZE)


if __name__ == "__main__":
    main()
