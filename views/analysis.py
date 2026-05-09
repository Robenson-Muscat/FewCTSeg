import streamlit as st
from streamlit_pdf_viewer import pdf_viewer
import requests

st.title("Method")

st.markdown("""
This section presents the winning approach.  
""")
#You can browse the slides interactively below.

#st.success("🏆 1st place")



pdf_url = st.secrets["pdf"]["url"]

response = requests.get(pdf_url)

if response.status_code == 200:
    pdf_viewer(response.content)
else:
    st.error("Loading error")
#pdf_viewer(pdf_url)

