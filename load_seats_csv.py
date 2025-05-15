#Teile des vorliegenden Codes wurden mit ChatGPT generiert oder modifiziert, OpenAI (2025)
import sqlite3
import csv

def readfile(conn, cursor, csv_file, building, floor):
    with open(csv_file, 'r', newline='', encoding='utf-8') as file:
        # CSV-Reader erstellen
        csv_reader = csv.reader(file, delimiter=';')  # Tabulator als Trennzeichen anpassen falls nötig
        
        # Kopfzeile überspringen (falls vorhanden)
        header = next(csv_reader)
        
        # Daten einfügen
        insert_query = """
        INSERT INTO seats (
            Building,
            Floor, 
            Nr,
            Steh_Sitz,
            Art,
            Position,
            Fenster,
            Steckdose,
            Sitzgelegenheit,
            Ort,
            Koordinaten_Y,
            Koordinaten_X
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """
        
        for row in csv_reader:
            # Konvertierung der Koordinaten zu Integer
            print(f"row:{building}, {floor}, {row[0]}, {row[1]}, {row[2]}, {row[3]}, {row[4]}, {row[5]}, {row[6]}, {row[7]}, {row[8]}, {row[9]}")
            converted_row = [
                building,             # Building    
                floor,        # Floor
                int(row[0]),        # Nr
                row[1],             # Steh/Sitz
                row[2],             # Art
                row[3],             # Position
                row[4].lower(),     # Fenster (einheitlich klein)
                row[5].lower(),     # Steckdose
                row[6].lower(),     # Sitzgelegenheit
                row[7],             # Ort
                int(row[8]),        # Koordinaten Y
                int(row[9])         # Koordinaten X
            ]
            print(f"Row inserted: {converted_row}")
            cursor.execute(insert_query, converted_row)
    # Transaktion bestätigen und Verbindung schließen
    conn.commit()

# Verbindung zur SQLite-Datenbank herstellen (wird erstellt, falls nicht vorhanden)
conn = sqlite3.connect('users.db')
cursor = conn.cursor()

# Tabelle erstellen (falls nicht vorhanden)
create_table_query = """
CREATE TABLE IF NOT EXISTS seats (
    Building TEXT,
    Floor INTEGER,
    Nr INTEGER,
    Steh_Sitz TEXT,
    Art TEXT,
    Position TEXT,
    Fenster TEXT,
    Steckdose TEXT,
    Sitzgelegenheit TEXT,
    Ort TEXT,
    Koordinaten_Y INTEGER,
    Koordinaten_X INTEGER,
    PRIMARY KEY (Building, Floor, Nr)  -- Zusammengesetzter Schlüssel
);"""
cursor.execute(create_table_query)
conn.commit()
print("Tabelle seats erstellt oder existiert bereits.")
# CSV-Daten importieren
csv_file = '../../../OneDrive/Desktop/Nest/Seats_Parterre_2025_05_11.csv'
readfile(conn, cursor, csv_file, 'Bibliothek', 0)
print(f"loaded: {csv_file}")
csv_file = '../../../OneDrive/Desktop/Nest/Seats_1st_floor_2025_05_11.csv'
readfile(conn, cursor, csv_file, 'Bibliothek', 1)    
print(f"loaded: {csv_file}")

# Transaktion bestätigen und Verbindung schließen
conn.commit()
conn.close()

print("Daten erfolgreich importiert!")