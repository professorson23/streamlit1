import streamlit as st

# Define the pages
main_page = st.Page("pages/p1_main.py", title="Main Page")
page_2 = st.Page("pages/p2_manual_predictor.py", title="Manual Predictor")
page_3 = st.Page("pages/p3_RTA.py", title="Real Time Dashboard")

# Set up navigation
pg = st.navigation([main_page,page_2,page_3])

# Run the selected page
pg.run()