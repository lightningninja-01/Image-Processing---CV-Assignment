import os
from pathlib import Path

import cv2
import numpy as np

# Keep matplotlib cache in project folder to avoid permission issues.
PROJECT_DIR = Path(__file__).resolve().parent
os.environ.setdefault("MPLCONFIGDIR", str(PROJECT_DIR / ".mplconfig"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

image_path = PROJECT_DIR / "download.webp"
img = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)

if img is None:
    raise FileNotFoundError(f"Could not load image: {image_path}")

# Contrast stretching.
min_val, max_val = np.min(img), np.max(img)
if max_val == min_val:
    cs_image = np.zeros_like(img)
else:
    cs_image = ((img - min_val) / (max_val - min_val) * 255).astype(np.uint8)

# Histogram equalization.
he_img = cv2.equalizeHist(img)

plt.figure(figsize=(10, 4))
plt.subplot(1, 3, 1)
plt.imshow(img, cmap="gray")
plt.title("Original Image")
plt.axis("off")

plt.subplot(1, 3, 2)
plt.imshow(cs_image, cmap="gray")
plt.title("Contrast Stretched Image")
plt.axis("off")

plt.subplot(1, 3, 3)
plt.imshow(he_img, cmap="gray")
plt.title("Histogram Equalized Image")
plt.axis("off")

plt.tight_layout()
output_path = PROJECT_DIR / "ip_output.png"
plt.savefig(output_path, dpi=150)
print(f"Saved result to: {output_path}")
