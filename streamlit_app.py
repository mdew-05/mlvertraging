import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime
df_all = pd.read_csv("df_all.csv", parse_dates=["start_time"])
# =========================
# Titel
# =========================
st.title("🚆 Vertraging voorspeller")

st.write("Vul de reiscontext in en krijg een voorspelling van de vertraging (≤ 20 min).")

# =========================
# Inputvelden
# =========================
ns_line = st.selectbox(
    "NS-lijn",
    options=df_all['ns_lines'].dropna().unique()
)

rdt_line = st.selectbox(
    "RDT-lijn",
    options=df_all['rdt_lines'].dropna().unique()
)

cause_group = st.selectbox(
    "Oorzaak (groep)",
    options=df_all['cause_group'].dropna().unique()
)

cause_nl = st.selectbox(
    "Specifieke oorzaak",
    options=df_all['cause_nl'].dropna().unique()
)

date = st.date_input("Datum van de reis")
time = st.time_input("Starttijd van de reis")

# =========================
# Feature engineering
# =========================
start_datetime = datetime.combine(date, time)

input_df = pd.DataFrame(
    [{
        'ns_lines': ns_line,
        'rdt_lines': rdt_line,
        'cause_group': cause_group,
        'cause_nl': cause_nl,
        'start_hour': start_datetime.hour,
        'start_dayofweek': start_datetime.weekday(),
        'start_month': start_datetime.month
    }]
)

# =========================
# Predict knop
# =========================
if st.button("🔮 Voorspel vertraging"):
    prediction = model.predict(input_df)[0]
    st.success(f"⏱️ Verwachte vertraging: **{prediction:.1f} minuten**")
