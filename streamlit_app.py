import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import streamlit as st
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error

# DF inlezen
df = pd.read_csv("df_all.csv", parse_dates=["start_time"])

# Titel

st.title("🚆 Vertraging voorspeller")

st.write("Vul de reiscontext in en krijg een voorspelling van de vertraging.")

ns_line = df['ns_lines'].mode()[0]        # most frequent ns_line
cause_group = df['cause_group'].mode()[0] # most frequent cause_group
cause_nl = df['cause_nl'].mode()[0]       # most frequent cause_nl
# Inputvelden
rdt_line = st.selectbox(
    "RDT-lijn",
    options=df['rdt_lines'].dropna().unique()
)
date = st.date_input("Datum van de reis")
time = st.time_input("Starttijd van de reis")

# Feature engineering
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
# 3. Features & target
# =========================
y = df['duration_minutes']
X = df[[ 'ns_lines', 'rdt_lines', 'cause_group', 'cause_nl', 'start_hour', 'start_dayofweek', 'start_month']]

# =========================
# 4. Preprocessing
# =========================
categorical_features = ['ns_lines', 'rdt_lines', 'cause_group', 'cause_nl']

numeric_features = ['start_hour', 'start_dayofweek', 'start_month']

preprocessor = ColumnTransformer(
    transformers=[
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features),
        ('num', 'passthrough', numeric_features)
    ]
)

# =========================
# 5. Model
# =========================
model = Pipeline(
    steps=[
        ('preprocessor', preprocessor),
        ('regressor', RandomForestRegressor(
            n_estimators=200,
            random_state=42,
            n_jobs=-1
        ))
    ]
)

# =========================
# 6. Train / test & evaluatie
# =========================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model.fit(X_train, y_train)

y_pred = model.predict(X_test)

mae = mean_absolute_error(y_test, y_pred)

# =========================
# Predict knop
# =========================
st.button("🔮 Voorspel vertraging")
prediction = model.predict(input_df)[0]
st.success(f"⏱️ Verwachte vertraging: **{prediction:.1f} minuten**")
