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
# 2. Feature engineering
# =========================
df['start_hour'] = df['start_time'].dt.hour
df['start_dayofweek'] = df['start_time'].dt.weekday
df['start_month'] = df['start_time'].dt.month

# Splits stations in lijstjes
df['station_list'] = df['rdt_station_names'].str.split(',')

# =========================
# 3. Inputvelden
# =========================
rdt_line = st.selectbox("RDT-lijn", options=df['rdt_lines'].unique())

# Stations op deze lijn
stations_line = sorted({s for sublist in df[df['rdt_lines'] == rdt_line]['station_list'] for s in sublist})
begin_station = st.selectbox("Beginstation", options=stations_line)

# Eindstations: alle stations behalve beginstation
end_stations = [s for s in stations_line if s != begin_station]
end_station = st.selectbox("Eindstation", options=end_stations)

# Datum en tijd
date = st.date_input("Datum van de reis")
time = st.time_input("Starttijd van de reis")
start_datetime = datetime.combine(date, time)

# =========================
# 4. Historische gemiddelde vertraging
# =========================
# Voor alle mogelijke begin-eind combinaties, per lijn en uur
def compute_avg_delay(df):
    records = []
    for _, row in df.iterrows():
        line = row['rdt_lines']
        hour = row['start_hour']
        stations = row['station_list']
        for i, b in enumerate(stations[:-1]):
            for e in stations[i+1:]:
                records.append({
                    'rdt_lines': line,
                    'begin_station': b,
                    'end_station': e,
                    'start_hour': hour,
                    'duration_minutes': row['duration_minutes']
                })
    return pd.DataFrame(records)

avg_df = compute_avg_delay(df)
avg_delay = avg_df.groupby(['rdt_lines','begin_station','end_station','start_hour'])['duration_minutes'].mean().reset_index()
avg_delay = avg_delay.rename(columns={'duration_minutes':'avg_delay'})

# Input dataframe
input_df = pd.DataFrame([{
    'rdt_lines': rdt_line,
    'begin_station': begin_station,
    'end_station': end_station,
    'start_hour': start_datetime.hour,
    'start_dayofweek': start_datetime.weekday(),
    'start_month': start_datetime.month
}])

# Voeg historisch gemiddelde toe
avg = avg_delay[
    (avg_delay['rdt_lines'] == rdt_line) &
    (avg_delay['begin_station'] == begin_station) &
    (avg_delay['end_station'] == end_station) &
    (avg_delay['start_hour'] == start_datetime.hour)
]
input_df['avg_delay'] = avg['avg_delay'].values[0] if not avg.empty else df['duration_minutes'].mean()

# =========================
# 5. Features & target
# =========================
y = df['duration_minutes']
X = pd.DataFrame({
    'rdt_lines': df['rdt_lines'],
    'begin_station': df['station_list'].apply(lambda x: x[0]),
    'end_station': df['station_list'].apply(lambda x: x[-1]),
    'start_hour': df['start_hour'],
    'start_dayofweek': df['start_dayofweek'],
    'start_month': df['start_month'],
    'avg_delay': df['duration_minutes']  # tijdelijk als proxy
})

# =========================
# 6. Preprocessing & model
# =========================
categorical_features = ['rdt_lines','begin_station','end_station']
numeric_features = ['start_hour','start_dayofweek','start_month','avg_delay']

preprocessor = ColumnTransformer(transformers=[
    ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features),
    ('num', 'passthrough', numeric_features)
])

model = Pipeline([
    ('preprocessor', preprocessor),
    ('regressor', RandomForestRegressor(n_estimators=200, random_state=42, n_jobs=-1))
])

# =========================
# 7. Train/test split
# =========================
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
mae = mean_absolute_error(y_test, y_pred)
st.write(f"✅ Model MAE op testset: {mae:.1f} minuten")

# =========================
# 8. Voorspelling
# =========================
if st.button("🔮 Voorspel vertraging"):
    prediction = model.predict(input_df)[0]
    st.success(f"⏱️ Verwachte vertraging: **{prediction:.1f} minuten**")
