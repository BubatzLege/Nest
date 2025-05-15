# 📚 Library Seat Reservation App

A simple Streamlit-based library seat reservation system with persistent user login, role-based access (Admin/Student), and room/time management.

---

## 🚀 Features

- 📅 Book and cancel library seat reservations
- 👥 Persistent login and user registration (SQLite)
- 🔐 Role-based: Students vs Admins
- 🛠 Admin dashboard with all reservations
- 📦 Exports reservations to .ics calendar
- 🖥 Responsive and modular interface

---

## 🛠 Installation Instructions

### 1. Install Miniconda (if you haven't already)

Download: https://docs.conda.io/en/latest/miniconda.html

### 2. Create environment with Python 3.12

```bash
conda create -n reserve python=3.12 -y
conda activate reserve
```

### 3. Install dependencies

```bash
pip install streamlit pandas ics
pip install tensorflow pandas numpy scikit-learn joblib
pip install plotly
```

---

## 🧾 Usage

1.a. Run the app:
```bash
streamlit run app.py
```

REMOTE ÜBER GITHUB VS CODE:
Local URL: http://localhost:8501

1.b. Run forecast prototype app
streamlit run forecast_prototype_app.py

2. Open in browser (usually at `http://localhost:8501`)


Also:
- Python Programme: GenerateReservations.py, learnSeatPredictionCriteria2.py, load_seats_csv.py 
  ausführen mit:
   python load_seats_csv.py (nur bei neuer Datenbank user.db)
   python GenerateReservations.py (zusätzliche Daten fürs Trainieren.)
   python learnSeatPredictionCriteria2.py (KI Modell neu Berechnen mit neuen Reservationen)
- Streamlit apps:
  app.py ==> die Nest app
  forecast_prototype_app.py ==> nur die KI-Vorhersage
  start mit: streamlit run app.py
  oder:      streamlit run forecast_prototype_app.py (Proof of concept vor Einbau in app.py als Zwischenschritt für Vorhersagen mit KI-Modell)
- KI-Modell:
  lstm_reservation_model.keras
  encoders.pkl
  user_encoder.pkl
  (pkl sind Data-Preprocessor-Funktionen, keras ist das KI-Modell)
- Bilder:
    Nest_LOGO.jpg
    Bib0.jpg (Bibliothek Floor 0)
    Bib1.jpg (Bibliothek Floor 1)
    Bib0_overlay.png (Temporäre Datei)
    Bib1_overlay.png (Temporäre Datei)
- Sitzplatzdaten
    Sitzplätze_Beschreibung_Bibliothek.xlsx
    Seats_1st_floor_2025_05_11.csv
    Seats_Parterre_2025_05_11.csv

---

## 🗃 Data

- `users.db`: stores users and reservations
- `app.py`: main Streamlit application
- `README.md`: this file

---

## 👤 Roles

- **Student**: can register, book, and cancel their own reservations
- **Admin**: can see all users and all reservations, and make any booking


---

## 💡 Notes

- Admin and user info stored in `users.db`
- Works offline and persists reservations

## user.db
- email: student@example.com
- password: password123
- Press "login" instead of pressing "Enter" on the keyboard

- email: demo@example.com
- password: 123
- email: admin@example.com
- password: 123

## create db seats in sqlite3 with table seats
- python load_seats_csv.py
- requires Seats_1st_floor_2025_05_11.csv und Seats_Parterre_2025_05_11.csv

## Bib1.jpg und Bib0.jpg are required for the display of images

## Nest_LOGO.jpg is required for the startup screen


Notizen zu KI

Was bedeutet df[features].values.reshape(1, 3, -1).astype('float32')

Der Ausdruck df[features].values.reshape(1, 3, -1).astype('float32') wird typischerweise in einem Machine Learning / Deep Learning Kontext verwendet. Er besteht aus mehreren Schritten, die ich dir hier im Detail erkläre:

1. df[features]
df ist ein Pandas-DataFrame.

features ist eine Liste von Spaltennamen (Strings), z. B. features = ['col1', 'col2', 'col3'].

df[features] gibt einen neuen DataFrame mit nur diesen Spalten.

2. .values
Gibt die zugrundeliegende NumPy-Array-Darstellung der Daten zurück.

Ergebnis ist ein NumPy-Array der Form (n_rows, n_features).

3. .reshape(1, 3, -1)
Formt das Array um in die Form (1, 3, x), wobei x automatisch bestimmt wird.

Beispiel:

Angenommen, df[features] enthält insgesamt 3 * n Spalten, dann wird das ganze Array als eine 3D-Struktur interpretiert.

reshape(1, 3, -1) bedeutet:

1: Batch-Größe (z. B. für ein einzelnes Sample)

3: Anzahl „Einheiten“ oder Zeitpunkte, Kanäle etc. (je nach Modell)

-1: wird automatisch berechnet, so dass die Gesamtanzahl der Elemente gleich bleibt.

4. .astype('float32')
Wandelt das Array in den Datentyp float32 um, was in Deep-Learning-Frameworks wie TensorFlow oder PyTorch Standard ist (Speichereffizienz, Kompatibilität mit GPU-Berechnungen).
