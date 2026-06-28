import sqlite3, json
conn = sqlite3.connect('data/normal/experiment_store.db')
cur = conn.cursor()
cur.execute("SELECT params FROM configs WHERE params LIKE '%connection_params%'")
print(cur.fetchone()[0])
conn.close()