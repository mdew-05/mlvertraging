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
from PIL import Image

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

tab1, tab2, tab3, tab4= st.tabs(["Introductie", "Datavisualisatie", "Model", "Evaluatie Model"])
with tab1:
    st.title("Introductie")
    st.write("Veelvoorkomend en vervelend probleem -> vertraging")
    st.write("Ongeveer 10% van reizigers meer dan 5 minuten (Nederlandse Spoorwegen, 2025)")
    st.write("Oplossing -> Machine learning model")
    st.write("Hypothese: Op drukkere plekken is de vertraging hoger")
    st.subheader("Praktische toepassingen")
with tab2:
    st.title("Datavisualisatie")
    st.write("De data is afkomstig van de rijden de treinen treinstoringen dataset")
    st.write("De data is opgeschoond en gefilterd")
    st.subheader("Duur van de storingen")
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
    
    st.subheader("Aantal storingen per uur van de dag")

    fig2, ax2 = plt.subplots()

    storingen_per_uur = (
        df["start_hour"]
        .value_counts()
        .sort_index()
    )

    ax2.bar(storingen_per_uur.index, storingen_per_uur.values)
    ax2.set_xlabel("Uur van de dag")
    ax2.set_ylabel("Aantal storingen")
    ax2.set_title("Drukste uren (storingen per start_hour)")
    ax2.set_xticks(range(0, 24))
    ax2.set_xticklabels(range(0, 24))
    st.pyplot(fig2)

    st.subheader("Drukste trajecten (meeste storingen)")

    # Tel aantal storingen per traject
    traject_counts = df["rdt_lines"].value_counts()
    
    # (optioneel) filter ruis, bv. minimaal 5 storingen
    traject_counts = traject_counts[traject_counts >= 5]
    
    # Top 10 drukste trajecten
    drukste_trajecten = traject_counts.nlargest(10)
    
    # Plot
    fig5, ax5 = plt.subplots(figsize=(8, 5))
    ax5.barh(drukste_trajecten.index, drukste_trajecten.values)
    ax5.set_xlabel("Aantal storingen")
    ax5.set_ylabel("Traject")
    ax5.set_title("Top 10 drukste trajecten")
    ax5.invert_yaxis()  # grootste bovenaan
    
    st.pyplot(fig5)

    st.subheader("Drukste stations (meeste storingen)")
    
    # Begin + eindstation combineren en tellen
    stations = pd.concat([
        df["begin_station"],
        df["end_station"]
    ])
    
    station_counts = stations.value_counts()
    
    # ruis filteren
    station_counts = station_counts[station_counts >= 10]
    
    # Drukste stations
    drukste_stations = station_counts.nlargest(10)

    fig3, ax3 = plt.subplots()
    ax3.barh(drukste_stations.index, drukste_stations.values)
    ax3.set_xlabel("Aantal storingen")
    ax3.set_ylabel("Station")
    ax3.set_title("Top 10 drukste stations")
    ax3.invert_yaxis()
    
    st.pyplot(fig3)
    

    #  Minst drukke stations
    st.subheader("Minst drukke stations (minste storingen)")
    
    minst_drukke_stations = (
        station_counts
        .nsmallest(10)
        .sort_values()
    )
    
    fig4, ax4 = plt.subplots()
    ax4.barh(minst_drukke_stations.index, minst_drukke_stations.values)
    ax4.set_xlabel("Aantal storingen")
    ax4.set_ylabel("Station")
    ax4.set_title("Top 10 minst drukke stations")
    
    st.pyplot(fig4)

            
with tab3: 
    st.title("🚆 Vertraging voorspeller")
    st.write("Vul de reiscontext in en krijg een voorspelling van de vertraging.")
    

    
    # 2. Inputvelden
    
    # Slider voor maximale duur
    max_delay = st.slider(
        "Maximale duur van de vertraging (minuten)",
        min_value=0,  
        max_value=300, 
        value=30,     
        step=5      
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

    # 4 Features & target
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
    
    # 5 Preprocessing
    categorical_features = ['ns_lines', 'rdt_lines', 'begin_station', 'end_station', 'cause_group', 'cause_nl']
    numeric_features = ['start_hour', 'start_dayofweek', 'start_month']
    
    preprocessor = ColumnTransformer(
        transformers=[
            ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features),
            ('num', 'passthrough', numeric_features)
        ]
    )
    
    # 6 Model 
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
    st.title("Evaluatie van model")
    image = Image.open("metrics ml.png")
    st.image(image, caption="Metrics van het model", use_column_width=True)
    st.subheader("Hypothese testen")
    st.write("Bij deze twee voorspellingen zijn alleen de stations veranderd, de maximale vertraging is ingesteld op 60 minuten, op 8:00 in de ochtend en op een vrijdag")
    col1, col2 = st.columns(2)
    with col1: 
        st.write("Schiphol airport naar Utrecht cen. (Drukke stations)")
        image1 = Image.open("schiphol naar utrecht.png")
        st.image(image1)
    with col2: 
        st.write("Breukelen naar Maarssen (Rustigere stations)")
        image2 = Image.open("breukelen naar maarssen.png")
        st.image(image2)

    st.subheader("Spits vs buiten spits")
    col3, col4 = st.columns(2)

    with col3: 
        st.write("Zwolle naar emmen om 17:00")
        image3 = Image.open("zwolle naar emmen om 17.png")
        st.image(image3)
    with col4: 
        st.write("Zwolle naar emmen om 2:00")
        image4 = Image.open("zwolle naar emmen om 2.png")
        st.image(image4)
        
    st.subheader("Doordeweeks vs weekend")
    col5, col6 = st.columns(2)

    with col5: 
        st.write("Deventer naar zutphen op vrijdag")
        image5 = Image.open("deventer naar zutphen doordeweeks.png")
        st.image(image5)
    with col6: 
        st.write("Deventer naar zutphen op zondag")
        image6 = Image.open("deventer naar zutphen weekend.png")
        st.image(image6)

