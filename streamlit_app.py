import streamlit as st




# --- PAGE SETUP ---
about_page = st.Page(
    page = "views/about_me.py",
    title = "About Me",
    icon = ":material/account_circle:",
    default = True,
)

presentation_page = st.Page(
    page="views/presentation.py",
    title="Project Overview",
    icon=":material/description:",
)

#project_1_page = st.Page( page = "views/images.py", title = "CT Scan images",  icon = ":material/image:")

project_2_page = st.Page(
    page = "views/analysis.py",
    title = "Method",
    icon = ":material/bar_chart:",
)

pg = st.navigation(
    {
        "Info": [about_page],
        "FewCTSeg Project": [presentation_page,project_2_page],# project_1_page, project_2_page],
    }
)


pg.run()



