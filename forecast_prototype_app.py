# streamlit_app.py
# Teile des vorliegenden Codes wurden mit ChatGPT generiert oder modifiziert OpenAI (2025)
import streamlit as st
import pandas as pd
import numpy as np
import sqlite3
from tensorflow import keras
import joblib

# Am Anfang der Streamlit-App (nur EIN Initialisierungsblock)
if 'model' not in st.session_state:
    # 1. Modell laden
    st.session_state.model = keras.models.load_model('lstm_reservation_model.keras')
    
    # 2. Encoder laden und strukturieren
    encoders = joblib.load("encoders.pkl")
    user_encoder = joblib.load("user_encoder.pkl")
    
    # 3. Korrekte Session State Struktur
    st.session_state.encoders = {
        'user_encoder': user_encoder,  # Direkter Zugriff
        'category_encoders': encoders   # Gesamtes Kategorie-Dict
    }
    # print("user_endcoder:", user_encoder)
    # print("encoders:", encoders)


    # 4. Sicherheitsprüfung
    required_keys = ['Steh_Sitz', 'Art', 'Position', 'Fenster', 
                    'Steckdose', 'Sitzgelegenheit', 'Ort']
    for key in required_keys:
        # print("key:", key)
        # Überprüfen, ob der Encoder existiert
        if key not in st.session_state.encoders['category_encoders']['encoders']:
            raise KeyError(f"Fehlender Encoder: {key}")

print ("model und encoders geladen")
# Datenbankverbindungen
def get_db_connection(db_name):
    return sqlite3.connect(f'{db_name}.db')

# Holt die letzten 3 Reservationen eines Benutzers
def get_last_reservations(username):
    conn = get_db_connection('users')
    query = f"""
    SELECT r.*, s.Steh_Sitz, s.Art, s.Position, s.Fenster, s.Steckdose, 
           s.Sitzgelegenheit, s.Ort 
    FROM reservations r
    JOIN seats s ON r.seat = 'Bibliothek / ' || s.Nr || ' / Stock: ' || s.Floor
    WHERE r.username = '{username}'
    ORDER BY r.date DESC, r.start_time DESC
    LIMIT 3
    """
    df = pd.read_sql(query, conn)
    conn.close()
    return df

# Vorverarbeitung der Daten
def preprocess_input(df, user_encoder, encoders):
    # Padding für weniger als 3 Reservationen
    if len(df) < 3:
        empty_rows = pd.DataFrame([[-1]*len(df.columns)]*(3-len(df)), columns=df.columns)
        df = pd.concat([df, empty_rows], ignore_index=True)
    
    # Zeitkonvertierung
    def convert_time(time_str):
        try:
            # Entferne alle Nicht-Zahlen und fülle auf 4 Stellen auf
            cleaned = ''.join(filter(str.isdigit, str(time_str))).zfill(4)
            hours = int(cleaned[:2])
            minutes = int(cleaned[2:4])
            return hours * 60 + minutes
        except:
            return -1  # oder np.nan für fehlende Werte

    # Anwendung auf die Spalte
    df['start_time'] = df['start_time'].apply(convert_time)
    # df['start_time'] = df['start_time'].str[:2].astype(int)*60 + df['start_time'].str[2:].astype(int)
    # df['dauer'] = df['end_time'].str[:2].astype(int)*60 + df['end_time'].str[2:].astype(int) - df['start_time']
    df['dauer'] = df['end_time'].apply(convert_time) - df['start_time'].apply(convert_time)
    # print("start_time:", df['start_time'], df['dauer'])
    # Encodings
    df['user_id'] = user_encoder.transform([df.iloc[0]['username']]*3)
    # print("user_id:", df['user_id'], df.iloc[0]['username'])
    cat_cols = ['Steh_Sitz','Art','Position','Fenster','Steckdose','Sitzgelegenheit','Ort']
    for col in cat_cols:
        df[col] = df[col].fillna('-1')
        df[col+'_enc'] = encoders['encoders'][col].transform(df[col].astype(str))
    
    # Feature-Auswahl
    features = ['user_id', 'start_time', 'dauer'] + [f'{col}_enc' for col in cat_cols]
    return df[features].values.reshape(1, 3, -1).astype('float32')

# Vorhersagefunktion

def predict_next():
    if 'username' not in st.session_state:
        st.error("Bitte zuerst einloggen")
        return
    
    df = get_last_reservations(st.session_state.username)
    if df.empty:
        st.warning("Keine bisherigen Reservationen gefunden")
        return
    
    # Zugriff auf die Encoder
    user_encoder = st.session_state.encoders['user_encoder']
    category_encoders = st.session_state.encoders['category_encoders']
    # print("Category encoders:", category_encoders)
    # print("Category_encoders encoders:", category_encoders['encoders'])
    
    # Preprocessing mit den korrekten Encodern
    input_seq = preprocess_input(df, user_encoder, category_encoders)
    
    # Vorhersage
    prediction = st.session_state.model.predict(input_seq)[0]
    
    # Dekodierung
    decoded = {}
    decoded['username'] = st.session_state.username
    decoded['start_time'] = f"{int(prediction[1]//60):02d}{int(prediction[1]%60):02d}"
    decoded['dauer'] = int(prediction[2])
    
    cat_cols = ['Steh_Sitz','Art','Position','Fenster','Steckdose','Sitzgelegenheit','Ort']
    # print("wichtig 2: encoders['category_encoders']['encoders']:", st.session_state.encoders['category_encoders']['encoders'])	
    for i, col in enumerate(cat_cols, start=3):
        decoded[col] = st.session_state.encoders['category_encoders']['encoders'][col].inverse_transform(
            [int(round(prediction[i]))])[0]
    
    # Ergebnisdarstellung
    st.subheader("Vorhergesagte nächste Reservation:")
    cols = st.columns(3)
    cols[0].metric("Startzeit", decoded['start_time'][:2]+":"+decoded['start_time'][2:])
    cols[1].metric("Dauer (min)", decoded['dauer'])
    cols[2].metric("Sitztyp", decoded['Steh_Sitz'])
    
    with st.expander("Details der Vorhersage"):
        st.write(f"**Art:** {decoded['Art']}")
        st.write(f"**Position:** {decoded['Position']}")
        st.write(f"**Fenster:** {decoded['Fenster']}")
        st.write(f"**Steckdose:** {decoded['Steckdose']}")
        st.write(f"**Sitzgelegenheit:** {decoded['Sitzgelegenheit']}")
        st.write(f"**Ort:** {decoded['Ort']}")

# UI-Komponenten
st.title("📍 Vorhersage nächster Sitzplatz")
# st.session_state.username = "lukas.schäfer2@student.unisg.ch"
# print("Title gesetzt session state gesetzt", st.session_state)
if 'username' in st.session_state:
    # st.write(f"Eingeloggt als: {st.session_state.username}")
    # if st.button("Vorhersage berechnen"):
    print("Vorhersage berechnen")
    predict_next()
    #print("Vorhersage nicht berechnet!")
else:
    st.text_input("Benutzername", key='username')
    st.button("Einloggen")