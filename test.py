import sqlite3
from pathlib import Path

conn = sqlite3.connect("dataset/sonicmorph.db")
cur = conn.cursor()

cur.execute("""
SELECT song_id,file_path
FROM songs
""")

missing = []

for song_id,path in cur.fetchall():

    if path and not Path(path).exists():
        missing.append(song_id)

for song_id in missing:

    cur.execute(
        "DELETE FROM songs WHERE song_id=?",
        (song_id,)
    )

print("Deleted missing:", len(missing))

conn.commit()
conn.close()