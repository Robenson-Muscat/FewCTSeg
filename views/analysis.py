import streamlit as st
from streamlit_pdf_viewer import pdf_viewer

st.title("Method")

st.markdown("""
This section presents the winning approach.  
You can browse the slides interactively below.
""")

pdf_path = "assets/method.pdf"

pdf_viewer(pdf_path)