import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, root_mean_squared_error, r2_score

# 1. Data inlezen
jaren = ["24", "25"]

dfs = [] 

for jaar in jaren:
    df = pd.read_csv(f"disruptions-20{jaar}.csv")

    # 1.1 Opschonen
    df = df.dropna(subset=['duration_minutes'])

    # 1.2 Tijdfeatures en stations maken
    df['start_time'] = pd.to_datetime(df['start_time'])
    df['start_hour'] = df['start_time'].dt.hour
    df['start_dayofweek'] = df['start_time'].dt.dayofweek
    df['start_month'] = df['start_time'].dt.month
    df['year'] = df['start_time'].dt.year
    df['begin_station'] = df['rdt_station_names'].str.split(',').str[0]
    df['end_station'] = df['rdt_station_names'].str.split(',').str[-1]
    dfs.append(df)

# 1.3 Alles samenvoegen
df = pd.concat(dfs, ignore_index=True)

tab1, tab2, tab3, tab4= st.tabs(["Introductie", "Data", "Model", "Evaluatie Model"])
with tab1:
    st.title("Introductie")
    st.write("Probleem -> vertraging")
    st.write("Ongeveer 10% van reizigers meer dan 5 minuten (Times, 2025)")
    st.write("Oplossing -> Machine learning model")
    
with tab2:
    st.title("Data")
    st.write("De data is afkomstig van de rijden de treinen treinstoringen dataset")

    fig, ax = plt.subplots()
    
    ax.hist(
        df.loc[df["duration_minutes"] < 500, "duration_minutes"],
        bins=50
    )
    ax.set_xlim(left=0)
    ax.set_xlabel("Vertraging (minuten)")
    ax.set_ylabel("Aantal storingen")
    ax.set_title("Verdeling van storingen (< 500 minuten)")
    
    st.pyplot(fig)
    
with tab3: 
    st.title("🚆 Vertraging voorspeller")
    st.write("Vul de reiscontext in en krijg een voorspelling van de vertraging.")
    

    
    # 2. Inputvelden
    
    # Slider voor maximale duur
    max_delay = st.slider(
        "Maximale duur van de vertraging (minuten)",
        min_value=0,    # minimaal 0 minuten
        max_value=300,  # maximaal 300 minuten, kan je aanpassen
        value=30,      # standaardwaarde
        step=5          # stapgrootte
    )
    
    # Filter dataframe
    df = df[df['duration_minutes'] <= max_delay]
    df = df.sort_values("rdt_lines")
    
    
    rdt_line = st.selectbox(
        "Traject",
        options=df['rdt_lines'].dropna().unique()
    )
    
    # Stations op deze lijn
    stations_line = sorted({s for sublist in df[df['rdt_lines'] == rdt_line]['rdt_station_names'].str.split(',') for s in sublist})
    
    begin_station = st.selectbox("Beginstation", options=stations_line)
    
    end_stations = [s for s in stations_line if s != begin_station]
    end_station = st.selectbox("Eindstation", options=end_stations)
    
    # Datum en tijd
    date = st.date_input("Datum van de reis")
    time = st.time_input("Starttijd van de reis")
    start_datetime = datetime.combine(date, time)

    # 4. Features & target
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
    
    # 5. Preprocessing
    categorical_features = ['ns_lines', 'rdt_lines', 'begin_station', 'end_station', 'cause_group', 'cause_nl']
    numeric_features = ['start_hour', 'start_dayofweek', 'start_month']
    
    preprocessor = ColumnTransformer(
        transformers=[
            ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features),
            ('num', 'passthrough', numeric_features)
        ]
    )
    
    # 6. Model
    model = Pipeline([
        ('preprocessor', preprocessor),
        ('regressor', RandomForestRegressor(n_estimators=50, random_state=42, n_jobs=-1))
    ])
    
    # 7. Train / test split & evaluatie
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model.fit(X_train, y_train)
    
    y_pred = model.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    rmse = root_mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    st.write(f"Model MAE op testset: {mae:.1f} minuten")
    st.write(f"Model RMSE op testset: {rmse:.1f} minuten")
    st.write(f"Model R2 op testset: {r2:.2f}")
    
    if st.button("🔮 Voorspel vertraging"):
        # Maak een klein dataframe met alleen de geselecteerde input
        input_df = pd.DataFrame([{
            'ns_lines': df[df['rdt_lines'] == rdt_line]['ns_lines'].iloc[0],
            'rdt_lines': rdt_line,
            'begin_station': begin_station,
            'end_station': end_station,
            'cause_group': df['cause_group'].mode()[0], 
            'cause_nl': df['cause_nl'].mode()[0],      
            'start_hour': start_datetime.hour,
            'start_dayofweek': start_datetime.weekday(),
            'start_month': start_datetime.month
        }])
    
        prediction = model.predict(input_df)[0]
        st.success(f"⏱️ Verwachte vertraging: **{prediction:.1f} minuten** ± **{mae:.1f}**")

with tab4:
    # Gebruik session_state om te controleren of evaluatie al is gedaan
    if 'tab4_done' not in st.session_state:
        st.session_state.tab4_done = False

    # Alleen uitvoeren als tab4 geopend wordt
    if not st.session_state.tab4_done:
        st.session_state.tab4_done = True

        st.title("Evaluatie van model")

        max_delays = range(0, 151, 10)
        mae_scores, rmse_scores, r2_scores = [], [], []

        for md in max_delays:
            df_md = df[df['duration_minutes'] <= md]

            if len(df_md) < 100:
                mae_scores.append(None)
                rmse_scores.append(None)
                r2_scores.append(None)
                continue

            y_md = df_md['duration_minutes']
            X_md = df_md[[
                'ns_lines', 'rdt_lines', 'begin_station', 'end_station',
                'cause_group', 'cause_nl', 'start_hour', 'start_dayofweek', 'start_month'
            ]]

            X_train_md, X_test_md, y_train_md, y_test_md = train_test_split(X_md, y_md, test_size=0.2, random_state=42)
            model_pipeline.fit(X_train_md, y_train_md)
            y_pred_md = model_pipeline.predict(X_test_md)

            mae_scores.append(mean_absolute_error(y_test_md, y_pred_md))
            rmse_scores.append(np.sqrt(np.mean((y_test_md - y_pred_md)**2)))
            r2_scores.append(r2_score(y_test_md, y_pred_md))

        # Plot
        fig, ax1 = plt.subplots(figsize=(10, 6))
        ax1.plot(max_delays, mae_scores, label="MAE (minuten)", marker='o')
        ax1.plot(max_delays, rmse_scores, label="RMSE (minuten)", marker='o')
        ax1.set_xlabel("Maximale vertraging (minuten)")
        ax1.set_ylabel("Fout (minuten)")
        ax1.legend(loc="upper left")
        ax1.grid(True)

        ax2 = ax1.twinx()
        ax2.plot(max_delays, r2_scores, label="R²", color='green', marker='x')
        ax2.set_ylabel("R²")
        ax2.legend(loc="upper right")

        st.pyplot(fig)
