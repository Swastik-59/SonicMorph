import sqlite3

conn = sqlite3.connect("dataset/sonicmorph.db")
cur = conn.cursor()

cur.execute("""
SELECT song_id, title, file_path
FROM songs
LIMIT 10
""")

for row in cur.fetchall():
    print(row)

conn.close()