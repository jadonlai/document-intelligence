import psycopg2



conn = psycopg2.connect(
    datbase='embeddings_db',
    user='postgres',
    password='dba',
    host='localhost',
    port='5432'
)

cursor = conn.cursor()

cursor.execute("SELECT * FROM items ORDER BY embedding <-> '[3,1,2]' LIMIT 5;")

results = cursor.fetchall()
# Print results (if any)
for row in results:
    print(row)

# Close connections
cursor.close()
conn.close()
