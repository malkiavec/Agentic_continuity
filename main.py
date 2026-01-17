import sys
import os

sys.path.append(os.path.dirname(__file__))
import pandas as pd
import requests

from continuity_page import render_continuity_page
import pandas as pd
import requests

from continuity_page import render_continuity_pageimport pandas as pd
import requests

from continuity_page import render_continuity_page

API_URL = "https://data.ny.gov/resource/6nbc-h7bj.json"

@st.cache_data
def load_df_raw():
    r = requests.get(API_URL, timeout=30)
    r.raise_for_status()
    return pd.DataFrame(r.json())

st.title("Lotto 6/59 Continuity Dashboard")

df_raw = load_df_raw()
render_continuity_page(df_raw)