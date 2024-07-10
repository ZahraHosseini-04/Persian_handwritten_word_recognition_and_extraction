import pandas as pd
import sqlite3

df = pd.read_csv('fname.csv')

# database file name
database_name = "fname.db"

#connection to database
conn = sqlite3.connect(database_name)
cursor = conn.cursor()

#convert data to sqlite database
df.to_sql('fname', conn, if_exists='replace', index=False)


#save changes and close
conn.commit()
conn.close()

