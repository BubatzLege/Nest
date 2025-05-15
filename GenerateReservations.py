#Teile des vorliegenden Codes wurden mit ChatGPT generiert oder modifiziert, OpenAI (2025)
import sqlite3
import random
from datetime import datetime, timedelta

# Häufige deutsche Vornamen und Nachnamen
firstnames = ["Lukas", "Leon", "Paul", "Finn", "Elias", "Jonas", "Noah", "Felix", "Maximilian", "Ben"]
lastnames = ["Müller", "Schmidt", "Schneider", "Fischer", "Weber", "Meyer", "Wagner", "Becker", "Hoffmann", "Schäfer"]

def generate_username():
    firstname = random.choice(firstnames)
    lastname = random.choice(lastnames)
    number = random.randint(1, 3)
    return f"{firstname.lower()}.{lastname.lower()}{number}@student.unisg.ch"

def generate_seat():
    stock = random.choice([0, 1])
    max_seat = 187 if stock == 0 else 283
    seat_number = random.randint(1, max_seat)
    return f"Bibliothek / {seat_number} / Stock: {stock}"

def generate_times():
    # Startzeit zwischen 08:00 und 19:00 in 15-Minuten-Schritten
    start_minutes = random.randint(0, (19 - 8) * 4) * 15
    start_time = (datetime.strptime("08:00", "%H:%M") + timedelta(minutes=start_minutes)).strftime("%H:%M:00")
    
    # Endzeit mindestens 15 Minuten später, aber maximal bis 20:00
    min_end = datetime.strptime(start_time, "%H:%M:%S") + timedelta(minutes=15)
    max_end = datetime.strptime("20:00", "%H:%M")
    if min_end > max_end:
        min_end = max_end
    latest_possible = min(max_end, min_end + timedelta(hours=3))
    delta_minutes = random.randint(0, int((latest_possible - min_end).total_seconds() // 60 // 15)) * 15
    end_time = (min_end + timedelta(minutes=delta_minutes)).strftime("%H:%M:00")
    
    return start_time, end_time

def generate_date():
    # Zufälliges Datum innerhalb der nächsten 30 Tage
    today = datetime.today()
    future_date = today + timedelta(days=random.randint(0, 30))
    return future_date.strftime("%Y-%m-%d")

def create_table():
    conn = sqlite3.connect("reservations.db")
    cursor = conn.cursor()
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS reservations (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT NOT NULL,
            date TEXT NOT NULL,
            start_time TEXT NOT NULL,
            end_time TEXT NOT NULL,
            seat TEXT NOT NULL
        )
    ''')
    conn.commit()
    conn.close()

def insert_reservations(n):
    conn = sqlite3.connect("users.db")
    cursor = conn.cursor()
    for _ in range(n):
        username = generate_username()
        date = generate_date()
        start_time, end_time = generate_times()
        seat = generate_seat()
        cursor.execute('''
            INSERT INTO reservations (username, date, start_time, end_time, seat)
            VALUES (?, ?, ?, ?, ?)
        ''', (username, date, start_time, end_time, seat))
    conn.commit()
    conn.close()
    print(f"{n} Reservationen wurden erfolgreich eingefügt.")

if __name__ == "__main__":
    create_table()
    try:
        count = int(input("Wie viele Reservationen sollen generiert werden? "))
        insert_reservations(count)
    except ValueError:
        print("Bitte eine gültige Zahl eingeben.")
