import streamlit as st
import numpy as np
import pickle
import sqlite3
import json
import os
import pandas as pd

with open("model1.pkl", 'rb') as f:
    model = pickle.load(f)

DB_PATH = "predictions.db"

def initialize_db():
    conn = sqlite3.connect(DB_PATH)
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
    conn.close()

initialize_db()

st.title("Flower Species Prediction")

sepal_length = st.slider("Sepal Length (cm)", 0.0, 8.0)
sepal_width = st.slider("Sepal Width (cm)", 0.0, 8.0)
petal_length = st.slider("Petal Length (cm)", 0.0, 8.0)
petal_width = st.slider("Petal Width (cm)", 0.0, 8.0)

species = ['Setosa', 'Versicolor', 'Virginica']

if st.button("Predict"):
    input_data = np.array([[sepal_length, sepal_width, petal_length, petal_width]])
    prediction = model.predict(input_data)[0]
    predicted_species = species[prediction]

    st.success(f"Predicted Species: {predicted_species}")

    input_dict = {
        "sepal.length": sepal_length,
        "sepal.width": sepal_width,
        "petal.length": petal_length,
        "petal.width": petal_width
    }

    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute('''
        INSERT INTO predictions (input_data, prediction)
        VALUES (?, ?)
    ''', (json.dumps(input_dict), predicted_species))
    conn.commit()
    conn.close()
