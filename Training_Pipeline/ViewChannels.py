# -*- coding: utf-8 -*-
"""
Created on Wed Apr  8 21:54:14 2026

@author: cns-th-lab
"""

import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np

# Load your image (works best for standard formats like .png, .tif)
# If using a specific video format, you might need cv2 or tifffile instead
img_path = r"C:\Users\cns-th-lab\DeepLabCut_Projects\AllRatsBulky-AITapus-2026-04-08\labeled-data\198.2025-07-28.mov_0001\img004080.png"
img = mpimg.imread(img_path)

# Check the shape to ensure it's (Height, Width, 2)
print("Image shape:", img.shape)

# Plot the two channels side-by-side
fig, axes = plt.subplots(1, 2, figsize=(10, 5))

# Channel 0
axes[0].imshow(img[:, :, 0], cmap='gray')
axes[0].set_title("Channel 0")
axes[0].axis('off')

# Channel 1
axes[1].imshow(img[:, :, 1], cmap='gray')
axes[1].set_title("Channel 1")
axes[1].axis('off')

plt.tight_layout()
plt.show()