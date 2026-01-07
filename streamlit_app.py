import pandas as pd
import streamlit as st
from datetime import datetime, timedelta
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
# 4. Filter op daadwerkelijke treinen
# =========================
# Kies een tijdswindow ±30 minuten rond de gekozen starttijd
time_window = timedelta(minutes=30)
df_filtered = df[
    (df['rdt_lines'] == rdt_line) &
    (df['station_list'].apply(lambda x: x[0]) == begin_station) &
    (df['station_list'].apply(lambda x: x[-1]) == end_station) &
    (df['start_time'] >= (start_datetime - time_window)) &
    (df['start_time'] <= (start_datetime + time_window))
]

# Historische gemiddelde vertraging in dit window
if not df_filtered.empty:
    avg_delay_value = df_filtered['duration_minutes'].mean()
else:
    # fallback: gemiddelde van alle ritten op deze lijn
    df_line = df[df['rdt_lines'] == rdt_line]
    avg_delay_value = df_line['duration_minutes'].mean() if not df_line.empty else df['duration_minutes'].mean()

# =========================
# 5. Input dataframe voor voorspelling
# =========================
input_df = pd.DataFrame([{
    'rdt_lines': rdt_line,
    'begin_station': begin_station,
    'end_station': end_station,
    'start_hour': start_datetime.hour,
    'start_dayofweek': start_datetime.weekday(),
    'start_month': start_datetime.month,
    'avg_delay': avg_delay_value
}])

# =========================
# 6. Features & target
# =========================
y = df['duration_minutes']

# Training features: begin = eerste station, eind = laatste station
X = pd.DataFrame({
    'rdt_lines': df['rdt_lines'],
    'begin_station': df['station_list'].apply(lambda x: x[0]),
    'end_station': df['station_list'].apply(lambda x: x[-1]),
    'start_hour': df['start_hour'],
    'start_dayofweek': df['start_dayofweek'],
    'start_month': df['start_month'],
    'avg_delay': df['duration_minutes']  # tijdelijk als proxy, wordt later gemerged
})

# Bereken historische avg_delay per lijn + begin + eind + uur
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
avg_df = pd.DataFrame(records)
avg_delay = avg_df.groupby(['rdt_lines','begin_station','end_station','start_hour'])['duration_minutes'].mean().reset_index()
avg_delay = avg_delay.rename(columns={'duration_minutes':'avg_delay'})

# Merge avg_delay in X_train
X = X.merge(avg_delay, on=['rdt_lines','begin_station','end_station','start_hour'], how='left')
X['avg_delay'] = X['avg_delay'].fillna(avg_delay['avg_delay'].mean())

# =========================
# 7. Preprocessing & model
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
# 8. Train/test split
# =========================
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
mae = mean_absolute_error(y_test, y_pred)
st.write(f"✅ Model MAE op testset: {mae:.1f} minuten")

# =========================
# 9. Voorspelling
# =========================
if st.button("🔮 Voorspel vertraging"):
    prediction = model.predict(input_df)[0]
    st.success(f"⏱️ Verwachte vertraging: **{prediction:.1f} minuten**")
