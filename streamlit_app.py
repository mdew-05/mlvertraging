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
# Basis features
df['start_hour'] = df['start_time'].dt.hour
df['start_dayofweek'] = df['start_time'].dt.weekday
df['start_month'] = df['start_time'].dt.month

# Historisch gemiddelde vertraging per lijn & uur
avg_delay = df.groupby(['rdt_lines', 'start_hour'])['duration_minutes'].mean().reset_index()
avg_delay = avg_delay.rename(columns={'duration_minutes': 'avg_delay_line_hour'})
df = df.merge(avg_delay, on=['rdt_lines', 'start_hour'], how='left')

# =========================
# 3. Inputvelden
# =========================
rdt_line = st.selectbox("RDT-lijn", options=df['rdt_lines'].dropna().unique())
date = st.date_input("Datum van de reis")
time = st.time_input("Starttijd van de reis")

start_datetime = datetime.combine(date, time)
input_df = pd.DataFrame([{
    'rdt_lines': rdt_line,
    'start_hour': start_datetime.hour,
    'start_dayofweek': start_datetime.weekday(),
    'start_month': start_datetime.month
}])

# Voeg historische gemiddelde vertraging toe aan input
avg = avg_delay[(avg_delay['rdt_lines'] == rdt_line) & (avg_delay['start_hour'] == start_datetime.hour)]
input_df['avg_delay_line_hour'] = avg['avg_delay_line_hour'].values[0] if not avg.empty else df['duration_minutes'].mean()

# =========================
# 4. Features & target
# =========================
y = df['duration_minutes']
X = df[['rdt_lines', 'start_hour', 'start_dayofweek', 'start_month', 'avg_delay_line_hour']]

# =========================
# 5. Preprocessing
# =========================
categorical_features = ['rdt_lines']
numeric_features = ['start_hour', 'start_dayofweek', 'start_month', 'avg_delay_line_hour']

preprocessor = ColumnTransformer(transformers=[
    ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features),
    ('num', 'passthrough', numeric_features)
])

# =========================
# 6. Model pipeline
# =========================
model = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('regressor', RandomForestRegressor(n_estimators=200, random_state=42, n_jobs=-1))
])

# =========================
# 7. Train/test split
# =========================
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model.fit(X_train, y_train)

# Evaluatie (optioneel)
y_pred = model.predict(X_test)
mae = mean_absolute_error(y_test, y_pred)
st.write(f"✅ Model MAE op testset: {mae:.1f} minuten")

# =========================
# 8. Voorspelling
# =========================
if st.button("🔮 Voorspel vertraging"):
    prediction = model.predict(input_df)[0]
    st.success(f"⏱️ Verwachte vertraging: **{prediction:.1f} minuten**")
