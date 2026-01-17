import streamlit as st
import pandas as pd
import requests
from continuity_page import render_continuity_page

# Set page config at the very beginning
st.set_page_config(page_title="Lotto 6/59 Continuity Analysis", layout="wide")

API_URL = "https://data.ny.gov/resource/6nbc-h7bj.json"

@st.cache_data
def load_df_raw():
    try:
        r = requests.get(API_URL, timeout=30)
        r.raise_for_status()
        return pd.DataFrame(r.json())
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return pd.DataFrame()

def main():
    st.sidebar.title("Navigation")
    app_mode = st.sidebar.selectbox("Choose the app mode", ["Continuity Analysis", "Data Info"])

    df_raw = load_df_raw()
    
    if df_raw.empty:
        st.warning("No data available. Please check the API connection.")
        return

    if app_mode == "Continuity Analysis":
        render_continuity_page(df_raw)
    elif app_mode == "Data Info":
        st.header("Dataset Overview")
        st.write("This application analyzes historical lottery draws using continuity features.")
        st.dataframe(df_raw)

if __name__ == "__main__":
    main()
