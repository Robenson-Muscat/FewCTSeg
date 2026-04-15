import streamlit as st
import os
import math
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from PIL import Image
from matplotlib.colors import ListedColormap

from utils import alphanumeric_sort

# --- CONFIG ---
IMAGE_DIR = "data/test-images/"
PREDICTIONS_CSV = "submissions/test_predictions_kfold.csv"

COLS_PER_ROW = 4
IMAGES_PER_PAGE = 40
NUM_CLASSES = 55

st.title("CT Scan Images")

# --- LEGEND COLORS ---
base_cmap = plt.cm.get_cmap("tab20", NUM_CLASSES)
colors = base_cmap(np.arange(NUM_CLASSES))
colors[0] = [0, 0, 0, 1]  # background = black
seg_cmap = ListedColormap(colors)

# --- CACHE DATA ---
@st.cache_data
def load_image_paths(image_dir):
    paths = sorted(glob.glob(os.path.join(image_dir, "*.png")), key=alphanumeric_sort)
    return paths

@st.cache_data
def load_predictions(csv_path):
    # Shape: (n_images, 256, 256)
    masks = pd.read_csv(csv_path, index_col=0).T.values.reshape(-1, 256, 256).astype(np.uint8)
    return masks

@st.cache_data
def load_image(path):
    return Image.open(path)

image_paths = load_image_paths(IMAGE_DIR)
masks = load_predictions(PREDICTIONS_CSV)

total_images = len(image_paths)
total_pages = math.ceil(total_images / IMAGES_PER_PAGE)

if total_images == 0:
    st.warning("No images found in the dataset.")
    st.stop()

if len(masks) != total_images:
    st.warning(
        f"Warning: number of images ({total_images}) and number of masks ({len(masks)}) do not match."
    )

# --- PAGINATION UI ---
page = st.number_input(
    "Page",
    min_value=1,
    max_value=total_pages,
    step=1
)

st.caption(f"Page {page} / {total_pages} — {total_images} images in total")

# --- IMAGE SELECTION ---
start = (page - 1) * IMAGES_PER_PAGE
end = min(start + IMAGES_PER_PAGE, total_images)

page_paths = image_paths[start:end]
page_names = [os.path.basename(p) for p in page_paths]

selected_name = st.selectbox("Select an image to segment", page_names)
selected_index = page_names.index(selected_name) + start

# --- DISPLAY SELECTED IMAGE + MASK ---
img = np.array(load_image(image_paths[selected_index]).convert("RGB"))
pred_mask = masks[selected_index]
seg_masked = np.ma.masked_where(pred_mask == 0, pred_mask)

fig, axes = plt.subplots(1, 2, figsize=(12, 6))

axes[0].imshow(img)
axes[0].set_title("Original image")
axes[0].axis("off")

axes[1].imshow(img)
axes[1].imshow(
    seg_masked,
    cmap=seg_cmap,
    vmin=0,
    vmax=NUM_CLASSES - 1,
    alpha=0.5,
    interpolation="nearest"
)
axes[1].set_title("Predicted mask")
axes[1].axis("off")

st.pyplot(fig, clear_figure=True)

# --- OPTIONAL GRID BELOW ---
#st.subheader("Images on this page")

for i in range(0, len(page_paths), COLS_PER_ROW):
    cols = st.columns(COLS_PER_ROW)
    for col, img_path in zip(cols, page_paths[i:i + COLS_PER_ROW]):
        image = load_image(img_path)
        with col:
            st.image(image, use_container_width=True)
            st.caption(os.path.basename(img_path))