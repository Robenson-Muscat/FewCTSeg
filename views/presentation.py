import streamlit as st

st.title("FewCTSeg Project Overview")


st.markdown("""

## Context
CT scans provide highly precise 3D images of the human body (up to 0.5 mm resolution), making it possible to capture detailed human anatomy.

The objective of this project is to automatically segment anatomical structures and tumors from CT scans.  
In other words, the goal is to identify and isolate the different shapes visible in CT images.

---

## Goal

The challenge is to segment structures **without exhaustive annotations**.

The training dataset is composed of two types of images:

- **Partially annotated CT scans**:
  - Contain segmentation masks for some anatomical structures
  - Do not represent all possible structures or variations
  - Example: one scan may include liver and spleen annotations, while another only includes the spleen

- **Raw CT scans (unannotated)**:
  - Contain no segmentation labels
  - Used as additional data for unsupervised learning

---

## Test Data

The test dataset contains fully annotated images.  
The evaluation metric measures the model’s ability to correctly segment and distinguish all anatomical structures present in each scan.

---

""")


st.image(
    "images/raidium_2024_1.png",
    caption="Example of an abdominal CT scan with segmented structures"
)

st.link_button(
    "View the project on GitHub",
    "https://github.com/Robenson-Muscat/FewCTSeg"
)