#Teile des vorliegenden Codes wurden mit ChatGPT generiert oder modifiziert, OpenAI (2025)
import sqlite3
import pandas as pd
import numpy as np
import joblib
from datetime import datetime
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Masking
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import train_test_split

# -----------------------------
# 1. Daten vorbereiten
# -----------------------------

def extract_seat_info(seat_str):
    try:
        parts = seat_str.split("/")
        return ("Bibliothek", int(parts[2].replace("Stock:", "").strip()), int(parts[1].strip()))
    except:
        return (None, None, None)

conn = sqlite3.connect("users.db")
reservations_df = pd.read_sql("SELECT * FROM reservations", conn)
seats_df = pd.read_sql("SELECT * FROM seats", conn)
conn.close()

reservations_df[['Building', 'Floor', 'Nr']] = reservations_df['seat'].apply(
    lambda s: pd.Series(extract_seat_info(s))
)

full_df = reservations_df.merge(seats_df, on=["Building", "Floor", "Nr"], how="left")

# -----------------------------
# 2. Vorverarbeitung
# -----------------------------

full_df['start_dt'] = pd.to_datetime(full_df['date'] + " " + full_df['start_time'])
full_df['end_dt'] = pd.to_datetime(full_df['date'] + " " + full_df['end_time'])
full_df['duration'] = (full_df['end_dt'] - full_df['start_dt']).dt.total_seconds() / 60

# Codieren der Nutzer
user_encoder = LabelEncoder()
full_df['user_id'] = user_encoder.fit_transform(full_df['username'])

# Sortieren
full_df.sort_values(by=['user_id', 'start_dt'], inplace=True)

# Kategorische Felder
cat_cols = ['Steh_Sitz', 'Art', 'Position', 'Fenster', 'Steckdose', 'Sitzgelegenheit', 'Ort']
encoders = {col: LabelEncoder().fit(full_df[col].astype(str)) for col in cat_cols}
for col in cat_cols:
    full_df[col + "_enc"] = encoders[col].transform(full_df[col].astype(str))

# -----------------------------
# 3. Sequenzen erzeugen
# -----------------------------

def encode_features(row):
    return [
        row['user_id'],
        row['start_dt'].hour * 60 + row['start_dt'].minute,
        row['duration']
    ] + [row[col + "_enc"] for col in cat_cols]

sequences = []
targets = []

for uid, group in full_df.groupby("user_id"):
    if len(group) < 4: # 4. 3. 2. ==> letzte Reservation vorherzusagen
        continue
    rows = group.reset_index(drop=True) # wenn 1 .. n Reservation 1 bis 4, 2 bis 5, 3 bis 6, .. n  bis 3 bis n
    for i in range(len(rows) - 3):
        seq = [encode_features(rows.iloc[i + j]) for j in range(3)]
        target = encode_features(rows.iloc[i + 3])
        sequences.append(seq)
        targets.append(target)

X = np.array(sequences)  # Shape: (n_samples, 3, n_features)
y = np.array(targets)    # Shape: (n_samples, n_features)

# -----------------------------
# 4. Daten splitten: Training Set, Training Validation Set, Out-Of-Training Test-Set
# -----------------------------

X_train_val, X_test, y_train_val, y_test = train_test_split(X, y, test_size=0.1, random_state=42) # 10 % Training Valierung
X_train, X_val, y_train, y_val = train_test_split(X_train_val, y_train_val, test_size=0.1111, random_state=42)  # 10% von 90% = 11% (9/10 1/9 = 10%)

# -----------------------------
# 5. LSTM-Modell
# -----------------------------

model = Sequential([
    Masking(mask_value=-1.0, input_shape=(3, X.shape[2])),
    LSTM(units=256, return_sequences=False),
    Dropout(0.3),
    Dense(128, activation='relu'),
    Dropout(0.2),
    Dense(y.shape[1], activation='linear')
])

model.compile(optimizer=Adam(0.001), loss='mse')
model.summary()

# -----------------------------
# 6. Training
# -----------------------------

model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=500,
    batch_size=32
)

# -----------------------------
# 7. Modell & Encoder speichern
# -----------------------------

# model.save("lstm_reservation_model.h5") 
model.save("lstm_reservation_model.keras")  # NEUES FORMAT

joblib.dump(encoders, "encoders.pkl")
# Im Trainingsteil beim Speichern:
joblib.dump({
    'user_encoder': user_encoder,
    'encoders': encoders  # Hier könnte die Benennung angepasst werden
}, 'encoders.pkl')

joblib.dump(user_encoder, "user_encoder.pkl")

# -----------------------------
# 8. Vorhersagefunktion
# -----------------------------

from tensorflow.keras.models import load_model

model = load_model("lstm_reservation_model.keras")
encoders = joblib.load("encoders.pkl")
user_encoder = joblib.load("user_encoder.pkl")

def predict_next_reservation(last_3_df):
    last_3_df['start_dt'] = pd.to_datetime(last_3_df['date'] + " " + last_3_df['start_time'])
    last_3_df['end_dt'] = pd.to_datetime(last_3_df['date'] + " " + last_3_df['end_time'])
    last_3_df['duration'] = (last_3_df['end_dt'] - last_3_df['start_dt']).dt.total_seconds() / 60
    last_3_df['user_id'] = user_encoder.transform([last_3_df.iloc[0]['username']]*3)

    for col in cat_cols:
        last_3_df[col + "_enc"] = encoders[col].transform(last_3_df[col].astype(str))

    seq = np.array([encode_features(row) for _, row in last_3_df.iterrows()])
    pred = model.predict(np.array([seq]))[0]
    return pred


mse = model.evaluate(X_test, y_test)
print(f"Test-MSE: {mse:.2f}")
mse = model.evaluate(X_train_val, y_train_val)
print(f"Test-MSE: {mse:.2f}")

""" Code für Streamlit App
# streamlit_app.py
import streamlit as st
import pandas as pd

st.title("📍 Vorhersage nächster Sitzplatz")

uploaded = st.file_uploader("Letzte 3 Reservationen hochladen (CSV)", type="csv")
if uploaded:
    df = pd.read_csv(uploaded)
    result = predict_next_reservation(df)
    st.subheader("Vorhergesagte nächste Reservation:")
    st.write(result)
"""