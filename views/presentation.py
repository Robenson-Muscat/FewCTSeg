import streamlit as st

st.title("FewCTSeg Project Overview")

st.markdown("""
FewCTSeg is a project solution to the medical image segmentation [challenge](https://challengedata.ens.fr/participants/challenges/165/) co-organized by **Raidium** and **ENS Paris-Saclay**.

This project aims to automatically segment anatomical structures and tumors in CT scans using machine learning and deep learning techniques.

Computed Tomography (CT) scans are medical imaging techniques that use X-rays combined with computer processing to generate detailed cross-sectional images of the human body. 
Unlike standard X-ray images, CT scans provide volumetric (3D) information by stacking multiple 2D slices, allowing for precise visualization of internal organs, tissues, and pathological structures.

In this challenge, the data consists of **2D slices extracted from CT scans**, specifically in the ** axial plane**, which corresponds to horizontal cross-sections of the body. 
Each slice represents a thin section of anatomy, and the objective is to segment the different structures visible within each individual image.


I am the **winner of this challenge**.

""")

#st.image("images/ranking.png",caption="Challenge ranking")

st.markdown("""
---

## Goal

The goal is to segment anatomical structures **without exhaustive annotations**.

The training data is composed of two types of CT scans:

- **Partially annotated CT scans**
  - Contain segmentation masks for only some anatomical structures
  - Do not describe all structures visible in the image
  - Example: one scan may include the liver and spleen, while another may include only the spleen

- **Raw CT scans**
  - Contain no segmentation masks
  - Can be used as additional data for unsupervised or weakly supervised learning

The test set contains new images with the corresponding segmentations, and the task is to correctly separate and identify all structures present in each scan.
""")

st.image("images/raidium_2024_1.png",caption="Example of an abdominal CT scan with segmented structures")

st.markdown("""
---

## Evaluation Metric

This is a **multi-class segmentation problem**.  
The performance is measured using the **DICE Score**, computed **per class** and then averaged. A Dice score of **1** indicates perfect segmentation, while **0** indicates no overlap.

For a given class $c$, the Dice score is defined as:

$$
\\mathrm{DICE}_c = \\frac{2|P_c \\cap G_c|}{|P_c| + |G_c|}
$$

where:
- $P_c$ is the set of pixels predicted as class $c$
- $G_c$ is the ground-truth set for class $c$

The final score is the **mean Dice across all classes**:

$$
\\mathrm{DICE}_{\\text{}} = \\frac{1}{C} \\sum_{c=1}^{C} \\mathrm{DICE}_c
$$


---

## Links

""")

st.link_button(
    "Open the GitHub repository",
    "https://github.com/Robenson-Muscat/FewCTSeg"
)

st.page_link("views/analysis.py", label="Open the Method tab")




#st.link_button("View the project on GitHub", "https://github.com/Robenson-Muscat/FewCTSeg")