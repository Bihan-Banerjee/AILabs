from flask import Flask, render_template, request, jsonify, redirect, url_for
import pandas as pd
import pickle
import sqlite3
import json
from datetime import datetime

app = Flask(__name__)
with open(r'd:\EXTRA\AILabs\IRIS folder\model1.pkl', 'rb') as f:
    model = pickle.load(f)

DB_PATH = 'predictions.db'

def init_db():
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

init_db()

species = ['Setosa', 'Versicolor', 'Virginica']

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/batch')
def batch_prediction():
    return render_template('batch.html')

@app.route('/api/predict', methods=['POST'])
def api_predict():
    try:
        data = request.get_json()
        
        required_fields = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width']
        if not all(field in data for field in required_fields):
            return jsonify({'error': 'Missing required fields'}), 400
        
        input_data = [[
            float(data['sepal_length']),
            float(data['sepal_width']),
            float(data['petal_length']),
            float(data['petal_width'])
        ]]
        
        prediction = model.predict(input_data)[0]
        predicted_species = species[prediction]
        
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        cursor.execute('''
            INSERT INTO predictions (input_data, prediction)
            VALUES (?, ?)
        ''', (json.dumps(data), predicted_species))
        conn.commit()
        conn.close()
        
        return jsonify({
            'prediction': predicted_species,
            'prediction_code': int(prediction),
            'input_data': data
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/batch/predict', methods=['POST'])
def batch_predict():
    try:
        if 'file' not in request.files:
            return redirect(request.url)
        
        file = request.files['file']
        if file.filename == '':
            return redirect(request.url)
        
        if file and file.filename.endswith('.csv'):
            df = pd.read_csv(file)
            
            required_columns = ['sepal.length', 'sepal.width', 'petal.length', 'petal.width']
            if not all(col in df.columns for col in required_columns):
                return "CSV file must contain sepal.length, sepal.width, petal.length, petal.width columns", 400
            
            X = df[required_columns]
            predictions = model.predict(X)
            df['prediction'] = [species[p] for p in predictions]
            
            conn = sqlite3.connect(DB_PATH)
            cursor = conn.cursor()
            
            for _, row in df.iterrows():
                input_data = row.drop('prediction').to_dict()
                cursor.execute('''
                    INSERT INTO predictions (input_data, prediction)
                    VALUES (?, ?)
                ''', (json.dumps(input_data), row['prediction']))
            
            conn.commit()
            conn.close()
            
            results_html = df.to_html(classes='table table-striped', index=False)
            return render_template('results.html', results=results_html)
        
        return "Invalid file format. Please upload a CSV file.", 400
        
    except Exception as e:
        return str(e), 500

@app.route('/predictions')
def view_predictions():
    conn = sqlite3.connect(DB_PATH)
    df = pd.read_sql('SELECT * FROM predictions ORDER BY timestamp DESC LIMIT 100', conn)
    conn.close()
    
    df['input_data'] = df['input_data'].apply(json.loads)
    
    display_df = pd.DataFrame()
    for _, row in df.iterrows():
        input_data = row['input_data']
        display_row = {
            'timestamp': row['timestamp'],
            'prediction': row['prediction'],
            **input_data
        }
        display_df = display_df.append(display_row, ignore_index=True)
    
    return render_template('results.html', results=display_df.to_html(index=False))

if __name__ == '__main__':
    app.run(debug=True)