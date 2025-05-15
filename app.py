#Teile des vorliegenden Codes wurde mit  ChatGPT generiert oder modifiziert. OpenAI (2025)
import base64
import hashlib
import locale
import sqlite3
import math
from datetime import datetime, timedelta

import pandas as pd
import streamlit as st
from ics import Calendar, Event
from PIL import Image, ImageDraw

import time
import streamlit as st
import streamlit.components.v1 as components

# -------------------- KI -----------------
import streamlit as st
import pandas as pd
import numpy as np
import sqlite3
from tensorflow import keras
import joblib
import plotly.graph_objects as go

@st.cache_data(ttl=300)
def load_data(query:str) -> pd.DataFrame:
    return pd.read_sql(query, conn)

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
    df['dauer'] = df['end_time'].apply(convert_time) - df['start_time'].apply(convert_time)

    # Encodings
    df['user_id'] = user_encoder.transform([df.iloc[0]['username']]*3)
    
    cat_cols = ['Steh_Sitz','Art','Position','Fenster','Steckdose','Sitzgelegenheit','Ort']
    for col in cat_cols:
        df[col] = df[col].fillna('-1')
        df[col+'_enc'] = encoders['encoders'][col].transform(df[col].astype(str))
    
    # Feature-Auswahl
    features = ['user_id', 'start_time', 'dauer'] + [f'{col}_enc' for col in cat_cols]
    return df[features].values.reshape(1, 3, -1).astype('float32')

# Vorhersagefunktion

def predict_next():
    filtersettings = {}
    if 'username' not in st.session_state:
        st.error("Bitte zuerst einloggen")
        return filtersettings
    
    df = get_last_reservations(st.session_state.username)
    if df.empty:
        st.warning("Keine bisherigen Reservationen gefunden")
        return filtersettings
    
    # Zugriff auf die Encoder
    user_encoder = st.session_state.encoders['user_encoder']
    category_encoders = st.session_state.encoders['category_encoders']
    
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
    
    with st.expander("Details der Vorhersage"):
        st.subheader("Vorhergesagte nächste Reservation:")
        cols = st.columns(3)
        cols[0].metric("Startzeit", decoded['start_time'][:2]+":"+decoded['start_time'][2:])
        cols[1].metric("Dauer (min)", decoded['dauer'])
        cols[2].metric("Sitztyp", decoded['Steh_Sitz'])
        st.write(f"**Art:** {decoded['Art']}")
        st.write(f"**Position:** {decoded['Position']}")
        st.write(f"**Fenster:** {decoded['Fenster']}")
        st.write(f"**Steckdose:** {decoded['Steckdose']}")
        st.write(f"**Sitzgelegenheit:** {decoded['Sitzgelegenheit']}")
        st.write(f"**Ort:** {decoded['Ort']}")
    filtersettings = {}
    filtersettings['start_time'] = decoded['start_time'][:2]+":"+decoded['start_time'][2:]
    filtersettings['Dauer'] = decoded['dauer']
    filtersettings['Steh_Sitz'] = decoded['Steh_Sitz']
    filtersettings['Art'] = decoded['Art'] 
    filtersettings['Position'] = decoded['Position']    
    filtersettings['Fenster'] = decoded['Fenster']  
    filtersettings['Steckdose'] = decoded['Steckdose']  
    filtersettings['Sitzgelegenheit'] = decoded['Sitzgelegenheit']  
    filtersettings['Ort'] = decoded['Ort'] 
    print("filtersettings:", filtersettings)
    return filtersettings

# ---------- ENDE KI -----------------

# Maps of seats
SeatMaps = ["Bib0.jpg", "Bib1.jpg"]
# ---------- LOGO SPLASH ----------
if "splash_shown" not in st.session_state:
    st.session_state.splash_shown = False

if not st.session_state.splash_shown:
    st.image("Nest_LOGO.jpg", use_container_width=True)
    st.session_state.splash_shown = True
    time.sleep(1)
    st.rerun()
    # Hinweis: Das rerun sorgt dafür, dass nach dem ersten Rendern die App weiterläuft.
st.set_page_config(layout="wide")
# ---------- CONFIG ----------
try:
    locale.setlocale(locale.LC_TIME, "de_CH.UTF-8")
except:  # noqa: E722
    pass
# ---------- DATABASE ----------
DB_PATH = "../../../OneDrive/Desktop/Nest/users.db"
conn = sqlite3.connect(DB_PATH, check_same_thread=False)
c = conn.cursor()
c.execute("""CREATE TABLE IF NOT EXISTS users (
    email TEXT PRIMARY KEY,
    password TEXT NOT NULL,
    role TEXT NOT NULL,
    first_name TEXT,
    last_name TEXT
)""")
c.execute("""CREATE TABLE IF NOT EXISTS reservations (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    username TEXT NOT NULL,
    date TEXT NOT NULL,
    start_time TEXT NOT NULL,
    end_time TEXT NOT NULL,
    seat TEXT NOT NULL
)""")
conn.commit()
# ---------- CONFIG ----------
ROOMS_BY_FLOOR = {}
ALL_ROOMS = []

# Räume dynamisch aus der seats-Tabelle laden
for row in c.execute("SELECT DISTINCT Building, Floor, Nr FROM seats ORDER BY Building, Floor, Nr"):
    building = row[0]
    floor = str(row[1])
    nr = row[2]
    key = f"{building} / Stock: {floor}"
    seat_name = f"{building} / {nr} / Stock: {floor}"
    if key not in ROOMS_BY_FLOOR:
        ROOMS_BY_FLOOR[key] = []
    ROOMS_BY_FLOOR[key].append(seat_name)
    ALL_ROOMS.append(seat_name)

# print(f"ROOMS_BY_FLOOR: {ROOMS_BY_FLOOR}")
# print(f"All rooms: {ALL_ROOMS}")

# ---------- SESSION ----------
if "logged_in" not in st.session_state:
    st.session_state.logged_in = False
if "username" not in st.session_state:
    st.session_state.username = None
if "user_role" not in st.session_state:
    st.session_state.user_role = None
if "full_name" not in st.session_state:
    st.session_state.full_name = None
if "reservation_time" not in st.session_state:
    st.session_state.reservation_time = datetime.now().time()
if "reservation_date" not in st.session_state:
    st.session_state.reservation_date = datetime.now().date()
if "reservation_seat" not in st.session_state:
    st.session_state.reservation_seat = ALL_ROOMS[0]
if "reservation_duration" not in st.session_state:
    st.session_state.reservation_duration = 1
if "menu" not in st.session_state:
    st.session_state.menu = None

# ---------- AUTH ----------
def hash_password(password):
    return hashlib.sha256(password.encode()).hexdigest()
# ---------- FUNCTIONS ----------
def get_list_distinct_values(cursor, column):
    cursor.execute(f"SELECT DISTINCT {column} FROM seats")
    return [row[0] for row in cursor.fetchall()]
# Filter-Dictionary mit den Listen der distinct values
Filter = {}
for col in ["Floor", "Steh_Sitz", "Art", "Position", "Fenster", "Steckdose", "Sitzgelegenheit", "Ort"]:
    Filter[col] = get_list_distinct_values(c, col)
# print("Filter:", Filter)

def load_user_reservations(username):
    # Hole alle Reservierungen des Users
    df_res = pd.read_sql(
        """
        SELECT *
        FROM reservations
        WHERE username = ?
          AND datetime(date || ' ' || end_time) > datetime('now')
        ORDER BY date, start_time, end_time, seat
        """,
        conn,
        params=(username,)
    )
    # Hole alle Sitzdaten
    df_seats = pd.read_sql("SELECT * FROM seats", conn)
    # Füge Koordinaten hinzu
    df = pd.merge(
        df_res,
        df_seats[["Building", "Nr", "Floor", "Koordinaten_Y", "Koordinaten_X"]],
        left_on="seat",
        right_on=df_seats.apply(lambda row: f"{row['Building']} / {row['Nr']} / Stock: {row['Floor']}", axis=1),
        how="left"
    )
    return df.to_dict(orient="records")

def delete_reservation_by_id(res_id):
    c.execute("DELETE FROM reservations WHERE id = ?", (res_id,))
    conn.commit()
def group_reservations_by_seat(user_res):
    """
    Gibt ein Dictionary zurück:
    {
        seat: {
            "x": Koordinaten_X,
            "y": Koordinaten_Y,
            "reservations": [ {Termin1}, {Termin2}, ... ]
        },
        ...
    }
    """
    seat_dict = {}
    for res in user_res:
        seat = res["seat"]
        x = res.get("Koordinaten_X")
        y = res.get("Koordinaten_Y")
        if seat not in seat_dict:
            seat_dict[seat] = {
                "x": x,
                "y": y,
                "reservations": []
            }
        seat_dict[seat]["reservations"].append(res)
    return seat_dict
def get_reservations_date_time_duration(reservations): # process every list element and get date, start_time, end_time, duration
    reservationstimes = ""
    for res in reservations:
        # print(f"res: {res}")
        if len(reservationstimes) > 0:
            reservationstimes += ("; " )
        reservationstimes += res['date'] + " " + res['start_time'] + " - " + res['end_time']
    return reservationstimes

if not st.session_state.logged_in:
    auth_action = st.sidebar.radio("Authentifizierung", ["Login", "Registrieren"])

    if auth_action == "Login":
        email = st.sidebar.text_input("Studenten-E-Mail")
        password = st.sidebar.text_input("Passwort", type="password")
        if st.sidebar.button("Einloggen"):
            c.execute(
                "SELECT password, role, first_name, last_name FROM users WHERE email = ?",
                (email,),
            )
            result = c.fetchone()
            if result and hash_password(password) == result[0]:
                st.session_state.logged_in = True
                st.session_state.username = email
                st.session_state.user_role = result[1]
                st.session_state.full_name = f"{result[2]} {result[3]}"
                st.rerun()
            else:
                st.error(
                    "Login fehlgeschlagen. Bitte überprüfen Sie Ihre Anmeldedaten. EMail:" + email +" pw:" + password + "hash:" + hash_password(password) + "db pw hash:" + str(result)
                                                    )

    elif auth_action == "Registrieren":
        new_email = st.sidebar.text_input("Neue E-Mail")
        first_name = st.sidebar.text_input("Vorname")
        last_name = st.sidebar.text_input("Nachname")
        new_password = st.sidebar.text_input("Neues Passwort", type="password")
        role = st.sidebar.selectbox("Rolle", ["Student", "Admin"])
        if st.sidebar.button("Registrieren"):
            try:
                c.execute(
                    "INSERT INTO users (email, password, role, first_name, last_name) VALUES (?, ?, ?, ?, ?)",
                    (
                        new_email,
                        hash_password(new_password),
                        role,
                        first_name,
                        last_name,
                    ),
                )
                conn.commit()
                st.success("Registrierung erfolgreich. Sie können sich nun einloggen.")
                st.rerun()
            except sqlite3.IntegrityError:
                st.error("Diese E-Mail ist bereits registriert.")

if not st.session_state.logged_in:
    st.warning("Bitte melden Sie sich an, um fortzufahren.")
    st.stop()

# ---------- APP ----------
username = st.session_state.username
is_admin = st.session_state.user_role == "Admin"
full_name = st.session_state.full_name

st.sidebar.markdown(f"👤 Eingeloggt als: {full_name} ({st.session_state.user_role})")
if st.sidebar.button("🚪 Abmelden"):
    for key in ["logged_in", "username", "user_role", "full_name"]:
        st.session_state[key] = None
    st.session_state.logged_in = False
    st.rerun()

main_menu = ["Tisch reservieren", "Meine Reservierungen", "Reservierungsübersicht"]
if is_admin:
    main_menu.append("Admin Dashboard")
    main_menu.append("Nest Key Performance Indicators") 
# Menü immer anzeigen und Auswahl übernehmen
menu = st.sidebar.radio("Menü", main_menu, index=main_menu.index(st.session_state.get("menu")) if st.session_state.get("menu") in main_menu else 0)
st.session_state.menu = menu

if menu == "Admin Dashboard" and is_admin:      #Das Admin-dashboard wurde aus demonstrationszwecken optimiert um die Benutzbarkeit zu gewährleister
    st.header("📊 Admin Dashboard")

    # Benutzerliste (gecached)
    users = load_data("SELECT email, role, first_name, last_name FROM users")
    st.subheader("Benutzerliste")
    st.dataframe(users, use_container_width=True)

    # Reservierungen (gecached + Pagination)
    all_res = load_data("SELECT * FROM reservations ORDER BY date, start_time")
    page_size = 50
    max_page = (len(all_res) - 1) // page_size + 1
    page = st.number_input("Seite", 1, max_page, 1)
    start_idx = (page - 1) * page_size
    res_slice = all_res.iloc[start_idx : start_idx + page_size]

    st.subheader(f"Reservierungen (Seite {page}/{max_page})")
    st.dataframe(res_slice, use_container_width=True)

    #Stornieren-Button pro Zeile
    st.subheader("Reservierung löschen")
    for r in res_slice.itertuples():
        c1, c2 = st.columns([5, 1])

        # Linke Spalte
        c1.markdown(
            f"**ID {r.id}** | {r.username} | "
            f"{r.date} {r.start_time}–{r.end_time} | {r.seat}"
        )

        # Rechte Spalte: Stornieren-Button
        if c2.button("Stornieren", key=f"del_{r.id}"):
            c.execute("DELETE FROM reservations WHERE id = ?", (r.id,))
            conn.commit()
            st.success(f"Reservierung {r.id} gelöscht")
            st.experimental_rerun()





elif menu == "Tisch reservieren":
    st.header("Tisch reservieren")

    # Filter-UI für Sitzplatzwahl
    st.markdown("""
        <style>
        div[data-baseweb="select"] > div {
            background-color: #222 !important;
            color: #fff !important;
        }
        div[data-baseweb="select"] span {
            color: #fff !important;
        }
        div[data-baseweb="popover"] {
            background-color: #222 !important;
        }
        </style>
    """, unsafe_allow_html=True)

    # KI Vorhersage Start
    if st.session_state.username is not None:
        # KI Vorhersage Start
        filtersettings = predict_next()
        # KI Vorhersage Ende
        selected_filters = {}
        filter_cols = st.columns(len(Filter))
        for idx, (col, values) in enumerate(Filter.items()):
            # Wenn ein Wert für diesen Filter in filtersettings vorhanden ist, nutze ihn als Default
            if col in filtersettings and filtersettings[col] in values:
                default = [filtersettings[col]]
            else:
                default = values
            selected = filter_cols[idx].multiselect(f"{col}", values, default=default)
            selected_filters[col] = selected
    else:
        # KI Vorhersage Ende
        selected_filters = {}
        filter_cols = st.columns(len(Filter))
        for idx, (col, values) in enumerate(Filter.items()):
            selected = filter_cols[idx].multiselect(f"{col}", values, default=values)
            selected_filters[col] = selected

    # Räume nach Filter einschränken
    # ("Wichtig: selected_filters:", selected_filters)
    seats_df = pd.read_sql("SELECT * FROM seats", conn)
    for col, selected in selected_filters.items():
        if selected and len(selected) < len(Filter[col]):
            seats_df = seats_df[seats_df[col].isin(selected)]
    filtered_rooms = [
        f"{row['Building']} / {row['Nr']} / Stock: {row['Floor']}"
        for _, row in seats_df.iterrows()
    ]
    # print("Wichtig: filtered_rooms:", filtered_rooms)

    reservation_date = st.date_input("Datum auswählen", value=st.session_state.get("reservation_date", datetime.now().date()))
    if st.session_state.username is not None:
        reservation_time = st.time_input(
            "Startzeit",
            value=filtersettings['start_time']
        )    
        # duration = st.number_input("Dauer (Stunden)", 1, 8, st.session_state.get("reservation_duration", 1))
        duration = st.number_input("Dauer (Stunden)", 1, 8, math.ceil(filtersettings['Dauer']/60))
    else:
        reservation_time = st.time_input(
            "Startzeit", value=st.session_state.get("reservation_time", datetime.now().time())
        )
        duration = st.number_input("Dauer (Stunden)", 1, 8, st.session_state.get("reservation_duration", 1))
    st.session_state.reservation_time = reservation_time

    if filtered_rooms and len(filtered_rooms) > 0:
        seat_number = st.selectbox("Platznummer", filtered_rooms, index=0)
    else:
        seat_number = st.selectbox("Platznummer", ALL_ROOMS, index=ALL_ROOMS.index(st.session_state.get("reservation_seat", ALL_ROOMS[0])))

    if st.button("Reservieren"):
        end_time = (
            datetime.combine(datetime.today(), reservation_time)
            + timedelta(hours=duration)
        ).time()
        c.execute(
            """
            INSERT INTO reservations (username, date, start_time, end_time, seat)
            VALUES (?, ?, ?, ?, ?)
        """,
            (
                username,
                str(reservation_date),
                str(reservation_time),
                str(end_time),
                seat_number,
            ),
        )
        conn.commit()
        st.success(
            f"Reservierung erfolgreich: {reservation_date}, {reservation_time} - {end_time}, {seat_number}"
        )
    # Bilder anzeigen
    st.markdown("---")
    st.subheader("Sitzplan")
    for img in SeatMaps:
        st.image(img, width=1024)

elif menu == "Meine Reservierungen":
    st.header("Meine Reservierungen")
    user_res = load_user_reservations(username)

    # Tabellen-Header
    header_cols = st.columns([1, 2, 2, 2, 2, 3])
    titles      = ["Aktion","Datum","Startzeit","Endzeit","Dauer","Platz"]
    for col, title in zip(header_cols, titles):
        col.markdown(
            f"<p class='white header'>{title}</p>",
            unsafe_allow_html=True
        )

    # Tabellen-Zeilen mit Buttons
    for idx, res in enumerate(user_res):
        # Zeilenfarbe
        row_style = "background-color: #f9f9f9; color #000"
        row_cols = st.columns([1, 2, 2, 2, 2, 3])

        # Dauer berechnen
        start_dt = datetime.strptime(f"{res['date']} {res['start_time']}", "%Y-%m-%d %H:%M:%S")
        end_dt = datetime.strptime(f"{res['date']} {res['end_time']}", "%Y-%m-%d %H:%M:%S")
        duration = end_dt - start_dt
        hours, remainder = divmod(duration.seconds, 3600)
        minutes = remainder // 60
        duration_str = f"{hours}h {minutes}m" if minutes else f"{hours}h"

        # Button-Label bestimmen
        if (
            datetime.now().date().isoformat() == res["date"]
            and res["start_time"] <= datetime.now().time().isoformat() <= res["end_time"]
        ):
            button_label = "Frühzeitig beenden"
        elif datetime.now().date().isoformat() < res["date"]:
            button_label = "Stornieren"
        else:
            button_label = ""

        # Button in Aktionsspalte
        with row_cols[0]:
            if button_label:
                if st.button(button_label, key=f"action_{idx}"):
                    delete_reservation_by_id(res["id"])
                    st.warning("Reservierung wurde beendet." if button_label == "Frühzeitig beenden" else "Reservierung storniert.")
                    st.rerun()
            else:
                st.markdown(f"<div style='{row_style}'>&nbsp;</div>", unsafe_allow_html=True)


        row_cols[1].markdown(f"<div style='{row_style}'>{res['date']}</div>", unsafe_allow_html=True)
        row_cols[2].markdown(f"<div style='{row_style}'>{res['start_time'][:5]}</div>", unsafe_allow_html=True)
        row_cols[3].markdown(f"<div style='{row_style}'>{res['end_time'][:5]}</div>", unsafe_allow_html=True)
        row_cols[4].markdown(f"<div style='{row_style}'>{duration_str}</div>", unsafe_allow_html=True)
        row_cols[5].markdown(f"<div style='{row_style}'>{res['seat']}</div>", unsafe_allow_html=True)

    # Export-Button wie gehabt
    if st.button("Als Kalenderdatei exportieren"):
        calendar = Calendar()
        for res in user_res:
            e = Event()
            e.name = f"Bibliotheksplatz: {res['seat']}"
            e.begin = f"{res['date']}T{res['start_time']}"
            e.end = f"{res['date']}T{res['end_time']}"
            e.location = res["seat"]
            calendar.events.add(e)
        ics_content = str(calendar)
        b64 = base64.b64encode(ics_content.encode()).decode()
        href = f'<a href="data:text/calendar;base64,{b64}" download="meine_reservierungen.ics">ICS-Datei herunterladen</a>'
        st.markdown(href, unsafe_allow_html=True)

    # Bilder anzeigen
    st.markdown("---")
    st.subheader("Sitzplan")
    # Nach dem Anzeigen der Bilder:
    # 1. Lade die Sitzpläne als PIL-Image
    img0 = Image.open("../../../OneDrive/Desktop/Nest/Bib0.jpg").convert("RGBA")
    img1 = Image.open("../../../OneDrive/Desktop/Nest/Bib1.jpg").convert("RGBA")
    draw0 = ImageDraw.Draw(img0)
    draw1 = ImageDraw.Draw(img1)
    # Zielbreite für Anzeige
    display_width = 600
    w0, h0 = img0.size
    w1, h1 = img1.size
    scale0 = display_width / w0
    scale1 = display_width / w1

    # 2. Sammle HTML-Overlays für Tooltips
    overlays0 = []
    overlays1 = []

    seat_infos = group_reservations_by_seat(user_res)
    # print("Seat Infos:", seat_infos)
    # Iteriere über alle Reservierungen
    for seat, values in seat_infos.items():

        # Hole Koordinaten und Stockwerk
        try:
            y = int(values.get("y", 0))
            x = int(values.get("x", 0))
            floor = int(seat.split("Stock: ")[-1]) if "Stock:" in res["seat"] else int(res.get("Floor", 0))
            if isinstance(floor, str):
                floor = int(floor)
            # Skaliere Koordinaten
            if floor == 0:
                x_disp = int(x * scale0)
                y_disp = int(y * scale0)
                dot_size = int(10 * scale0)
                dot_radius = dot_size // 2
            elif floor == 1:
                x_disp = int(x * scale1)
                y_disp = int(y * scale1)
                dot_size = int(10 * scale1)
                dot_radius = dot_size // 2
            else:
                continue
            #print(f"Coordinates: {x}, {y}, floor: {floor} , {type(floor).__name__}, Termine: {values.get("reservations")}")
        except Exception:
            continue

        
        # Tooltip-Text
        tooltip = get_reservations_date_time_duration(values.get("reservations"))
        print(f"seat: {seat} floor: {floor}: Coordinates: {x}, {y}")
        # Zeichne Punkt und erstelle Overlay
        if floor == 0:
            draw0.ellipse((x-7, y-7, x+7, y+7), fill="red", outline="black", width=2)
            overlays0.append(
                f'<div class="dot" style="top:{y_disp-dot_radius}px;left:{x_disp-dot_radius}px;width:{dot_size}px;height:{dot_size}px;border-radius:{dot_radius}px;" title="{tooltip}"></div>'
            )
            # overlays0.append(f'<div class="dot" style="top:{y-7}px;left:{x-7}px;" title="{tooltip}"></div>')
        elif floor == 1:
            draw1.ellipse((x-7, y-7, x+7, y+7), fill="red", outline="black", width=2)
            overlays1.append(
                f'<div class="dot" style="top:{y_disp-dot_radius}px;left:{x_disp-dot_radius}px;width:{dot_size}px;height:{dot_size}px;border-radius:{dot_radius}px;" title="{tooltip}"></div>'
            )

    # 3. Speichere temporäre Bilder
    img0.save("Bib0_overlay.png")
    img1.save("Bib1_overlay.png")

    # 4. Zeige die Bilder mit HTML-Overlay (für Tooltip)
    def show_image_with_overlay(img_path, overlays, width=600):
        img = Image.open(img_path)
        w, h = img.size
        scale = width / w
        html = f"""
        <div style="position:relative;width:{width}px;height:{int(h*scale)}px;">
            <img src="data:image/png;base64,{base64.b64encode(open(img_path, 'rb').read()).decode()}" width="{width}px" style="display:block;">
            {''.join(overlays)}
        </div>
        <style>
        .dot {{
            position:absolute;
            background:red;
            border:2px solid black;
            opacity:0.8;
            cursor:pointer;
            z-index:10;
        }}
        .dot:hover {{
            opacity:1.0;
            box-shadow:0 0 10px #f00;
        }}
        </style>
        """
        components.html(html, height=int(h*scale)+10)

    st.markdown("---")
    st.subheader("Sitzplan mit Reservierungen")

    st.write("Parterre (Stock 0):")
    show_image_with_overlay("../../../OneDrive/Desktop/Nest/Bib0_overlay.png", overlays0, width=600)

    st.write("1. Stock (Stock 1):")
    show_image_with_overlay("../../../OneDrive/Desktop/Nest/Bib1_overlay.png", overlays1, width=600)
elif menu == "Reservierungsübersicht":

    #neu
    st.header("Reservierungsübersicht")

    # Tag-Auswahl und Zeitbereich ganz am Anfang!
    today = datetime.now().date()
    selected_day = st.selectbox(
        "Tag auswählen",
        [today + timedelta(days=i) for i in range(7)],
        format_func=lambda d: d.strftime("%A, %d. %B %Y"),
    )
    start_hour, end_hour = 8, 19    

    # CSS Filter
    st.markdown("""
        <style>
        /* Hintergrundfarbe für alle Multiselects und Selectboxen */
        div[data-baseweb="select"] > div {
            background-color: #222 !important;
            color: #fff !important;
        }
        /* Textfarbe der Dropdown-Optionen */
        div[data-baseweb="select"] span {
            color: #fff !important;
        }
        /* Hintergrund der Dropdown-Liste */
        div[data-baseweb="popover"] {
            background-color: #222 !important;
        }
        </style>
    """, unsafe_allow_html=True) 

    # --- Filter-UI ---
    selected_filters = {}
    filter_cols = st.columns(len(Filter))
    for idx, (col, values) in enumerate(Filter.items()):
        selected = filter_cols[idx].multiselect(f"{col}", values, default=values)
        selected_filters[col] = selected

    # --- Räume nach Filter einschränken ---
    # Hole alle Sitzdaten aus der seats-Tabelle
    seats_df = pd.read_sql("SELECT * FROM seats", conn)

    # Filter anwenden
    for col, selected in selected_filters.items():
        if selected and len(selected) < len(Filter[col]):
            seats_df = seats_df[seats_df[col].isin(selected)]

    # Erzeuge die gefilterte Raumliste im gleichen Format wie ALL_ROOMS
    filtered_rooms = [
        f"{row['Building']} / {row['Nr']} / Stock: {row['Floor']}"
        for _, row in seats_df.iterrows()
    ]

    # Wenn keine Räume übrig bleiben, Hinweis anzeigen
    if not filtered_rooms:
        st.warning("Keine Sitzplätze entsprechen den gewählten Filterkriterien.")
    else:
        # Restlicher Code: grid_data, df, Direktbuchung, Pivot etc.
        grid_data = []
        all_res = pd.read_sql(
            "SELECT * FROM reservations WHERE date = ?", conn, params=(str(selected_day),)
        )
        for room in filtered_rooms:
            for hour in range(start_hour, end_hour + 1):
                slot_time = datetime.combine(selected_day, datetime.min.time()) + timedelta(
                    hours=hour
                )
                status = "Verfügbar"
                for _, res in all_res.iterrows():
                    if (
                        res["seat"] == room
                        and res["start_time"]
                        <= slot_time.time().isoformat()
                        < res["end_time"]
                    ):
                        status = (
                            "Meine Reservierung"
                            if res["username"] == username
                            else "Reserviert"
                        )
                        break
                grid_data.append(
                    {
                        "Raum": room,
                        "Zeit": f"{hour:02d}:00",
                        "Status": status,
                        "Datetime": slot_time,
                    }
                )

        df = pd.DataFrame(grid_data)

        with st.expander("Direktbuchung (ein Slot klicken)"):
            col1, col2 = st.columns(2)
            for idx, row in df.iterrows():
                with col1 if idx % 2 == 0 else col2:
                    if row["Status"] == "Verfügbar" or is_admin:
                        if st.button(
                            f"Buchen: {row['Raum']} - {row['Zeit']}", key=f"book_{idx}"
                        ):
                            end_time = (
                                (row["Datetime"] + timedelta(hours=1)).time().isoformat()
                            )
                            c.execute(
                                """
                                INSERT INTO reservations (username, date, start_time, end_time, seat)
                                VALUES (?, ?, ?, ?, ?)
                            """,
                                (
                                    username,
                                    str(selected_day),
                                    row["Datetime"].time().isoformat(),
                                    end_time,
                                    row["Raum"],
                                ),
                            )
                            conn.commit()
                            st.success(
                                f"Reservierung erfolgreich für {row['Raum']} um {row['Zeit']}"
                            )
                            st.rerun()

        grid = df.pivot(index="Raum", columns="Zeit", values="Status")

        def color_map(val):
            return {
                "Verfügbar": "background-color: #d4edda; color: black;",
                "Reserviert": "background-color: #f8d7da; color: black;",
                "Meine Reservierung": "background-color: #c62121; color: black; font-weight: bold;",
            }.get(val, "")

        st.dataframe(grid.style.map(color_map), use_container_width=True, height=600)
        st.caption(
            "Farblegende: Verfügbar (grün), Reserviert (rot), Meine Reservierung (blau)"
        )
elif menu == "Nest Key Performance Indicators":
    st.header("Nest Key Performance Indicators")

    # Lade alle Reservierungen
    df = pd.read_sql("SELECT * FROM reservations", conn)

    # Berechne die Dauer jeder Reservierung in Minuten
    def get_duration(row):
        try:
            start = datetime.strptime(f"{row['date']} {row['start_time']}", "%Y-%m-%d %H:%M:%S")
            end = datetime.strptime(f"{row['date']} {row['end_time']}", "%Y-%m-%d %H:%M:%S")
            return (end - start).total_seconds() / 60
        except Exception:
            return None

    df["duration_min"] = df.apply(get_duration, axis=1)
    df = df.dropna(subset=["duration_min"])

    # Durchschnittliche Reservationsdauer (in Minuten und Stunden)
    avg_duration_min = df["duration_min"].mean()
    avg_duration_h = avg_duration_min / 60 if avg_duration_min else 0

    st.subheader("Durchschnittliche Reservationsdauer")
    st.metric("Ø Dauer (Minuten)", f"{avg_duration_min:.1f}")
    st.metric("Ø Dauer (Stunden)", f"{avg_duration_h:.2f}")

    # Durchschnittliche Reservationsdauer je Startstunde (08:00 bis 19:00)
    st.subheader("Durchschnittliche Reservationsdauer nach Startzeit (je volle Stunde)")
    df["start_hour"] = df["start_time"].str[:2].astype(int)
    hour_stats = []
    for hour in range(8, 20):
        mask = df["start_hour"] == hour
        avg = df.loc[mask, "duration_min"].mean()
        count = mask.sum()
        hour_stats.append({
            "Stunde": f"{hour:02d}:00",
            "Ø Dauer (Minuten)": avg if count > 0 else 0,
            "Anzahl Reservierungen": count
        })

    # DataFrame für Chart
    hour_df = pd.DataFrame(hour_stats)

    # Plotly Bar Chart
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=hour_df["Stunde"],
        y=hour_df["Ø Dauer (Minuten)"],
        name="Ø Dauer (Minuten)",
        marker_color="royalblue"
    ))
    fig.update_layout(
        title="Belegung nach Uhrzeit",
        xaxis_title="Startzeit",
        yaxis_title="Ø Dauer (Minuten)",
        bargap=0.2,
        height=500,
    )
    # KPIs als Text ins Chart einfügen
    fig.add_annotation(
        x=0.5, y=1.15, xref="paper", yref="paper",
        text=f"Ø Dauer (Minuten): <b>{avg_duration_min:.1f}</b><br>Ø Dauer (Stunden): <b>{avg_duration_h:.2f}</b>",
        showarrow=False,
        font=dict(size=16, color="black"),
        align="center",
        bgcolor="rgba(255,255,255,0.7)"
    )

    st.plotly_chart(fig, use_container_width=True)

    # Optional: Tabelle weiterhin anzeigen
    st.dataframe(
        hour_df.assign(**{"Ø Dauer (Minuten)": hour_df["Ø Dauer (Minuten)"].apply(lambda x: f"{x:.1f}" if x > 0 else "-")}),
        use_container_width=True,
        column_config={
            "Stunde": st.column_config.Column(width="8ch"),
            "Ø Dauer (Minuten)": st.column_config.Column(width="10ch"),
            "Anzahl Reservierungen": st.column_config.Column(width="10ch"),
        }
    )



