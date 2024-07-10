import pandas as pd
import sqlite3

df = pd.read_excel('lname.xlsx')

# database file name
database_name = "lname.db"



#connection to database
conn = sqlite3.connect(database_name)
cursor = conn.cursor()

#convert data to sqlite database
df.to_sql('lname', conn, if_exists='replace', index=False)

# #show database
# cursor.execute("SELECT * FROM lname")
# rows = cursor.fetchall()
# for row in rows:
#     print(row)

#save changes and close
conn.commit()
conn.close()










# import sqlite3

# # اتصال به پایگاه داده SQLite
# conn = sqlite3.connect('lname.db')
# cursor = conn.cursor()

# # خواندن اطلاعات از فایل txt
# with open('lname.txt', 'r') as file:
#     for line in file:
#         data = line.strip() # فرضاً داده‌ها با جداکننده , در فایل txt قرار دارند
#         cursor.execute("INSERT INTO lname (lname) VALUES (?)", (data[0]))

# # ذخیره تغییرات و بستن اتصال
# conn.commit()
# conn.close()
