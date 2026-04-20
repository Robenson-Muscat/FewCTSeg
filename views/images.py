import streamlit as st
import os
import math
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import requests
import zipfile

from io import BytesIO
from PIL import Image
from matplotlib.colors import ListedColormap

from utils import alphanumeric_sort

# =========================
# CONFIG
# =========================
DATA_URL = "https://challengedata.ens.fr/media/public/test-images.zip"
BASE_DIR = "data"
IMAGE_DIR = os.path.join(BASE_DIR, "test-images")

PREDICTIONS_CSV = "submissions/test_predictions_kfold_combined_1404_postproc_20_bestval_confidencefallback.csv"

COLS_PER_ROW = 5
IMAGES_PER_PAGE = 50
NUM_CLASSES = 55

st.title("CT Scan Images")

# =========================
# DOWNLOAD + EXTRACT
# =========================
@st.cache_resource
def download_and_extract(url, extract_dir):
    os.makedirs(extract_dir, exist_ok=True)

    # déjà extrait → on skip
    if os.path.exists(IMAGE_DIR) and len(os.listdir(IMAGE_DIR)) > 0:
        return

    st.info("Downloading dataset...")

    response = requests.get(url)
    if response.status_code != 200:
        st.error("Download failed")
        st.stop()

    zip_file = zipfile.ZipFile(BytesIO(response.content))
    zip_file.extractall(extract_dir)

    st.success("Dataset ready!")

download_and_extract(DATA_URL, BASE_DIR)

# =========================
# COLOR MAP
# =========================
base_cmap = plt.cm.get_cmap("tab20", NUM_CLASSES)
colors = base_cmap(np.arange(NUM_CLASSES))
colors[0] = [0, 0, 0, 1]
seg_cmap = ListedColormap(colors)

# =========================
# LOADERS (cached)
# =========================
@st.cache_data
def load_image_paths(image_dir):
    paths = sorted(
        glob.glob(os.path.join(image_dir, "*.png")),
        key=alphanumeric_sort
    )
    return paths

@st.cache_data
def load_predictions(csv_path):
    masks = pd.read_csv(csv_path, index_col=0).T.values
    masks = masks.reshape(-1, 256, 256).astype(np.uint8)
    return masks

@st.cache_data
def load_image(path):
    return Image.open(path).convert("RGB")

image_paths = load_image_paths(IMAGE_DIR)
masks = load_predictions(PREDICTIONS_CSV)

total_images = len(image_paths)
total_pages = math.ceil(total_images / IMAGES_PER_PAGE)

if total_images == 0:
    st.warning("No images found.")
    st.stop()

if len(masks) != total_images:
    st.warning(
        f"Mismatch: {total_images} images vs {len(masks)} masks"
    )

# =========================
# SESSION STATE (page memory)
# =========================
if "page" not in st.session_state:
    st.session_state.page = 1

page = st.number_input(
    "Page",
    min_value=1,
    max_value=total_pages,
    step=1,
    key="page"
)

st.caption(f"Page {page} / {total_pages} — {total_images} images")

# =========================
# PAGINATION
# =========================
start = (page - 1) * IMAGES_PER_PAGE
end = min(start + IMAGES_PER_PAGE, total_images)

page_paths = image_paths[start:end]
page_names = [os.path.basename(p) for p in page_paths]

# =========================
# IMAGE SELECTION
# =========================
selected_name = st.selectbox("Select image", page_names)
selected_index = page_names.index(selected_name) + start

# =========================
# DISPLAY SINGLE IMAGE + MASK
# =========================
img = np.array(load_image(image_paths[selected_index]))
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

# =========================
# GRID DISPLAY (FASTER)
# =========================
st.subheader("Images on this page")

for i in range(0, len(page_paths), COLS_PER_ROW):
    cols = st.columns(COLS_PER_ROW)

    for col, img_path in zip(cols, page_paths[i:i + COLS_PER_ROW]):
        with col:
            img = load_image(img_path)
            st.image(img, use_container_width=True)
            st.caption(os.path.basename(img_path))
