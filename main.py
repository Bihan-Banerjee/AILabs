import pandas as pd
import pickle
import sqlite3

df = pd.read_csv('iris.csv')

with open('model1.pkl', 'rb') as f:
    model = pickle.load(f)

X = df[['sepal.length', 'sepal.width', 'petal.length', 'petal.width']]
predictions = model.predict(X)
df['prediction'] = predictions

conn = sqlite3.connect('predictions.db')
cursor = conn.cursor()
cursor.execute('''
CREATE TABLE IF NOT EXISTS predictions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    input_data TEXT,
    prediction TEXT,
    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
)
''')
conn.commit()

for _, row in df.iterrows():
    input_data = row.drop('prediction').to_json()
    prediction = str(row['prediction'])
    cursor.execute('''
        INSERT INTO predictions (input_data, prediction)
        VALUES (?, ?)
    ''', (input_data, prediction))

conn.commit()
conn.close()
