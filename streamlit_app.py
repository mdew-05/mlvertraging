import pandas as pd
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

st.title("🚆 Vertraging voorspeller")
st.write("Vul de reiscontext in en krijg een voorspelling van de vertraging.")

# =========================
# 1. Input: RDT-lijn, beginstation, eindstation, datum/tijd
# =========================
rdt_line = st.selectbox(
    "RDT-lijn",
    options=df['rdt_lines'].dropna().unique()
)

# Stations van deze lijn
stations_line = sorted({s for sublist in df[df['rdt_lines'] == rdt_line]['rdt_station_names'].str.split(',') for s in sublist})

begin_station = st.selectbox("Beginstation", options=stations_line)

# Eindstations: alle stations behalve beginstation
end_stations = [s for s in stations_line if s != begin_station]
end_station = st.selectbox("Eindstation", options=end_stations)

# Datum en tijd
date = st.date_input("Datum van de reis")
time = st.time_input("Starttijd van de reis")
start_datetime = datetime.combine(date, time)

# =========================
# 2. Feature engineering
# =========================
ns_line = df['ns_lines'].mode()[0]        # meest voorkomende ns_line
cause_group = df['cause_group'].mode()[0] # meest voorkomende cause_group
cause_nl = df['cause_nl'].mode()[0]       # meest voorkomende cause_nl

input_df = pd.DataFrame([{
    'ns_lines': ns_line,
    'rdt_lines': rdt_line,
    'begin_station': begin_station,
    'end_station': end_station,
    'cause_group': cause_group,
    'cause_nl': cause_nl,
    'start_hour': start_datetime.hour,
    'start_dayofweek': start_datetime.weekday(),
    'start_month': start_datetime.month
}])

# =========================
# 3. Features & target
# =========================
y = df['duration_minutes']

X = df[[
    'ns_lines',
    'rdt_lines',
    'begin_station',  # toegevoegd
    'end_station',    # toegevoegd
    'cause_group',
    'cause_nl',
    'start_hour',
    'start_dayofweek',
    'start_month'
]]

# Voeg begin/end station toe aan df als eerste/laatste van lijst
X['begin_station'] = X['rdt_station_names'].str.split(',').str[0]
X['end_station'] = X['rdt_station_names'].str.split(',').str[-1]

# =========================
# 4. Preprocessing
# =========================
categorical_features = ['ns_lines', 'rdt_lines', 'begin_station', 'end_station', 'cause_group', 'cause_nl']
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
model = Pipeline([
    ('preprocessor', preprocessor),
    ('regressor', RandomForestRegressor(n_estimators=200, random_state=42, n_jobs=-1))
])

# =========================
# 6. Train / test & evaluatie
# =========================
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
mae = mean_absolute_error(y_test, y_pred)

# =========================
# 7. Voorspelling knop
# =========================
if st.button("🔮 Voorspel vertraging"):
    prediction = model.predict(input_df)[0]
    st.success(f"⏱️ Verwachte vertraging: **{prediction:.1f} minuten** ± **{mae:.1f}**")
