import pandas as pd
import streamlit as st
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error

# =========================
# 1. Data inlezen
# =========================
df = pd.read_csv("df_all.csv", parse_dates=["start_time"])

st.title("🚆 Vertraging voorspeller")
st.write("Vul de reiscontext in en krijg een voorspelling van de vertraging.")

# =========================
# 2. Feature engineering: tijd en stations
# =========================
df['start_hour'] = df['start_time'].dt.hour
df['start_dayofweek'] = df['start_time'].dt.weekday
df['start_month'] = df['start_time'].dt.month

# Voeg begin en eindstation toe
df['begin_station'] = df['rdt_station_names'].str.split(',').str[0]
df['end_station'] = df['rdt_station_names'].str.split(',').str[-1]

# =========================
# 3. Inputvelden
# =========================
rdt_line = st.selectbox(
    "Traject",
    options=df['rdt_lines'].dropna().unique()
)

# Stations op deze lijn
stations_line = sorted({s for sublist in df[df['rdt_lines'] == rdt_line]['rdt_station_names'].str.split(',') for s in sublist})

begin_station = st.selectbox("Beginstation", options=stations_line)

# Eindstations: alle stations behalve beginstation
end_stations = [s for s in stations_line if s != begin_station]
end_station = st.selectbox("Eindstation", options=end_stations)

# Datum en tijd
date = st.date_input("Datum van de reis")
time = st.time_input("Starttijd van de reis")
start_datetime = datetime.combine(date, time)

# Kies meest voorkomende ns_line, cause_group, cause_nl als default
ns_line = df['ns_lines'].mode()[0]
cause_group = df['cause_group'].mode()[0]
cause_nl = df['cause_nl'].mode()[0]

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
# 4. Features & target
# =========================
y = df['duration_minutes']

X = df[[
    'ns_lines',
    'rdt_lines',
    'begin_station',
    'end_station',
    'cause_group',
    'cause_nl',
    'start_hour',
    'start_dayofweek',
    'start_month'
]]

# =========================
# 5. Preprocessing
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
# 6. Model
# =========================
model = Pipeline([
    ('preprocessor', preprocessor),
    ('regressor', RandomForestRegressor(n_estimators=200, random_state=42, n_jobs=-1))
])

# =========================
# 7. Train / test split & evaluatie
# =========================
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
mae = mean_absolute_error(y_test, y_pred)
st.write(f"✅ Model MAE op testset: {mae:.1f} minuten")

# =========================
# 8. Predict knop
# =========================
if st.button("🔮 Voorspel vertraging"):
    prediction = model.predict(input_df)[0]
    st.success(f"⏱️ Verwachte vertraging: **{prediction:.1f} minuten** ± **{mae:.1f}**")
