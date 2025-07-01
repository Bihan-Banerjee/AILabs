# MLflow Flask App for Champion Model Management and Inference
# This app allows you to set champion models and perform inference via REST API

from flask import Flask, request, jsonify, render_template_string
import mlflow
import mlflow.sklearn
from mlflow.tracking import MlflowClient
import numpy as np
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
import json
import os
import warnings
warnings.filterwarnings('ignore')

# Initialize Flask app
app = Flask(__name__)

# Initialize MLflow client
client = MlflowClient()

# Global variables for champion model
CHAMPION_MODEL = None
CHAMPION_MODEL_NAME = None
IRIS_FEATURE_NAMES = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width']
IRIS_CLASS_NAMES = ['setosa', 'versicolor', 'virginica']

def setup_initial_models():
    """Create some initial models if they don't exist"""
    print("Setting up initial models...")
    
    # Set MLflow experiment
    mlflow.set_experiment("Iris_Flask_Demo")
    
    # Load iris dataset
    iris = load_iris()
    X = iris.data
    y = iris.target
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Create a few models with different configurations
    models_config = [
        ("Champion_LogisticRegression", LogisticRegression(C=1.0, random_state=42)),
        ("Champion_RandomForest", RandomForestClassifier(n_estimators=100, random_state=42)),
        ("Champion_SVM", SVC(C=1.0, kernel='rbf', random_state=42))
    ]
    
    for model_name, model in models_config:
        try:
            # Check if model already exists
            try:
                registered_model = client.get_registered_model(model_name)
                print(f"Model {model_name} already exists, skipping...")
                continue
            except:
                pass
            
            # Train model
            with mlflow.start_run(run_name=f"Training_{model_name}"):
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)
                accuracy = accuracy_score(y_test, y_pred)
                
                # Log metrics and parameters
                mlflow.log_metric("accuracy", accuracy)
                mlflow.log_params(model.get_params())
                
                # Log model
                model_info = mlflow.sklearn.log_model(
                    sk_model=model,
                    artifact_path="model",
                    registered_model_name=model_name
                )
                
                print(f"Registered model: {model_name} with accuracy: {accuracy:.4f}")
                
        except Exception as e:
            print(f"Error setting up model {model_name}: {e}")

def load_champion_model(model_name, version="latest"):
    """Load champion model from MLflow"""
    global CHAMPION_MODEL, CHAMPION_MODEL_NAME
    try:
        if version == "latest":
            model_uri = f"models:/{model_name}/latest"
        else:
            model_uri = f"models:/{model_name}/{version}"
            
        CHAMPION_MODEL = mlflow.sklearn.load_model(model_uri)
        CHAMPION_MODEL_NAME = f"{model_name}_v{version}"
        print(f"Loaded champion model: {CHAMPION_MODEL_NAME}")
        return True
    except Exception as e:
        print(f"Error loading champion model: {e}")
        return False

# HTML template for the web interface
HTML_TEMPLATE = """
<!DOCTYPE html>
<html>
<head>
    <title>MLflow Champion Model Interface</title>
    <style>
        body { font-family: Arial, sans-serif; margin: 40px; background-color: #f5f5f5; }
        .container { max-width: 800px; margin: 0 auto; background: white; padding: 30px; border-radius: 10px; box-shadow: 0 0 10px rgba(0,0,0,0.1); }
        h1 { color: #333; text-align: center; }
        .section { margin: 30px 0; padding: 20px; border: 1px solid #ddd; border-radius: 5px; }
        .form-group { margin: 15px 0; }
        label { display: block; margin-bottom: 5px; font-weight: bold; }
        input, select, textarea { width: 100%; padding: 8px; border: 1px solid #ccc; border-radius: 4px; }
        button { background-color: #007bff; color: white; padding: 10px 20px; border: none; border-radius: 4px; cursor: pointer; margin: 5px; }
        button:hover { background-color: #0056b3; }
        .result { background-color: #e8f5e8; padding: 15px; border-radius: 5px; margin-top: 15px; }
        .error { background-color: #ffe8e8; padding: 15px; border-radius: 5px; margin-top: 15px; }
        .champion-info { background-color: #e3f2fd; padding: 15px; border-radius: 5px; }
    </style>
</head>
<body>
    <div class="container">
        <h1>🏆 MLflow Champion Model Interface</h1>
        
        <!-- Champion Model Status -->
        <div class="section">
            <h3>Current Champion Model</h3>
            <div class="champion-info">
                <p><strong>Model:</strong> {{ champion_model or "No champion model set" }}</p>
                <p><strong>Status:</strong> {{ "Ready for inference" if champion_model else "Not ready" }}</p>
            </div>
        </div>
        
        <!-- Set Champion Model -->
        <div class="section">
            <h3>Set Champion Model</h3>
            <div class="form-group">
                <label>Model Name:</label>
                <select id="modelName">
                    <option value="Champion_LogisticRegression">Champion_LogisticRegression</option>
                    <option value="Champion_RandomForest">Champion_RandomForest</option>
                    <option value="Champion_SVM">Champion_SVM</option>
                </select>
            </div>
            <div class="form-group">
                <label>Version:</label>
                <input type="text" id="modelVersion" value="latest" placeholder="latest or version number">
            </div>
            <button onclick="setChampion()">Set as Champion</button>
        </div>
        
        <!-- Model Inference -->
        <div class="section">
            <h3>Model Inference</h3>
            <p>Enter Iris flower measurements for prediction:</p>
            
            <div class="form-group">
                <label>Sepal Length (cm):</label>
                <input type="number" id="sepalLength" step="0.1" placeholder="e.g., 5.1">
            </div>
            <div class="form-group">
                <label>Sepal Width (cm):</label>
                <input type="number" id="sepalWidth" step="0.1" placeholder="e.g., 3.5">
            </div>
            <div class="form-group">
                <label>Petal Length (cm):</label>
                <input type="number" id="petalLength" step="0.1" placeholder="e.g., 1.4">
            </div>
            <div class="form-group">
                <label>Petal Width (cm):</label>
                <input type="number" id="petalWidth" step="0.1" placeholder="e.g., 0.2">
            </div>
            
            <button onclick="predict()">Predict Iris Species</button>
            <button onclick="predictBatch()">Predict Batch (JSON)</button>
            
            <!-- JSON Input for Batch Prediction -->
            <div class="form-group" style="margin-top: 20px;">
                <label>JSON Input (for batch prediction):</label>
                <textarea id="jsonInput" rows="4" placeholder='{"instances": [{"sepal_length": 5.1, "sepal_width": 3.5, "petal_length": 1.4, "petal_width": 0.2}]}'></textarea>
            </div>
        </div>
        
        <!-- Results -->
        <div id="results"></div>
    </div>

    <script>
        function setChampion() {
            const modelName = document.getElementById('modelName').value;
            const version = document.getElementById('modelVersion').value;
            
            fetch('/set_champion', {
                method: 'POST',
                headers: {'Content-Type': 'application/json'},
                body: JSON.stringify({model_name: modelName, version: version})
            })
            .then(response => response.json())
            .then(data => {
                showResult(data, data.status === 'success');
                if (data.status === 'success') {
                    setTimeout(() => location.reload(), 1500);
                }
            })
            .catch(error => showResult({message: 'Error: ' + error}, false));
        }
        
        function predict() {
            const data = {
                sepal_length: parseFloat(document.getElementById('sepalLength').value),
                sepal_width: parseFloat(document.getElementById('sepalWidth').value),
                petal_length: parseFloat(document.getElementById('petalLength').value),
                petal_width: parseFloat(document.getElementById('petalWidth').value)
            };
            
            fetch('/predict', {
                method: 'POST',
                headers: {'Content-Type': 'application/json'},
                body: JSON.stringify(data)
            })
            .then(response => response.json())
            .then(data => showResult(data, data.status !== 'error'))
            .catch(error => showResult({message: 'Error: ' + error}, false));
        }
        
        function predictBatch() {
            const jsonInput = document.getElementById('jsonInput').value;
            
            try {
                const data = JSON.parse(jsonInput);
                fetch('/predict_batch', {
                    method: 'POST',
                    headers: {'Content-Type': 'application/json'},
                    body: JSON.stringify(data)
                })
                .then(response => response.json())
                .then(data => showResult(data, data.status !== 'error'))
                .catch(error => showResult({message: 'Error: ' + error}, false));
            } catch (e) {
                showResult({message: 'Invalid JSON format'}, false);
            }
        }
        
        function showResult(data, isSuccess) {
            const resultsDiv = document.getElementById('results');
            const className = isSuccess ? 'result' : 'error';
            resultsDiv.innerHTML = `<div class="${className}"><pre>${JSON.stringify(data, null, 2)}</pre></div>`;
        }
        
        // Load sample data
        window.onload = function() {
            document.getElementById('sepalLength').value = '5.1';
            document.getElementById('sepalWidth').value = '3.5';
            document.getElementById('petalLength').value = '1.4';
            document.getElementById('petalWidth').value = '0.2';
        }
    </script>
</body>
</html>
"""

@app.route('/')
def home():
    """Home page with web interface"""
    return render_template_string(HTML_TEMPLATE, champion_model=CHAMPION_MODEL_NAME)

@app.route('/set_champion', methods=['POST'])
def set_champion():
    """Set a model version as champion"""
    try:
        data = request.json
        model_name = data.get('model_name')
        version = data.get('version', 'latest')
        
        if not model_name:
            return jsonify({
                'status': 'error',
                'message': 'Model name is required'
            }), 400
        
        # Load the model as champion
        success = load_champion_model(model_name, version)
        
        if success:
            return jsonify({
                'status': 'success',
                'message': f'Model {model_name} version {version} set as champion',
                'champion_model': CHAMPION_MODEL_NAME
            })
        else:
            return jsonify({
                'status': 'error',
                'message': f'Failed to load model {model_name} version {version}'
            }), 500
            
    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': f'Error setting champion: {str(e)}'
        }), 500

@app.route('/predict', methods=['POST'])
def predict():
    """Perform inference on single input"""
    try:
        if CHAMPION_MODEL is None:
            return jsonify({
                'status': 'error',
                'message': 'No champion model set. Please set a champion model first.'
            }), 400
        
        data = request.json
        
        # Validate input
        required_features = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width']
        for feature in required_features:
            if feature not in data:
                return jsonify({
                    'status': 'error',
                    'message': f'Missing required feature: {feature}'
                }), 400
        
        # Prepare input for prediction
        input_features = np.array([[
            data['sepal_length'],
            data['sepal_width'],
            data['petal_length'],
            data['petal_width']
        ]])
        
        # Make prediction
        prediction = CHAMPION_MODEL.predict(input_features)[0]
        
        # Get prediction probabilities if available
        try:
            probabilities = CHAMPION_MODEL.predict_proba(input_features)[0]
            prob_dict = {IRIS_CLASS_NAMES[i]: float(probabilities[i]) for i in range(len(IRIS_CLASS_NAMES))}
        except:
            prob_dict = None
        
        response = {
            'status': 'success',
            'prediction': {
                'class': IRIS_CLASS_NAMES[prediction],
                'class_index': int(prediction),
                'probabilities': prob_dict
            },
            'input_features': data,
            'model_used': CHAMPION_MODEL_NAME
        }
        
        return jsonify(response)
        
    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': f'Prediction error: {str(e)}'
        }), 500

@app.route('/predict_batch', methods=['POST'])
def predict_batch():
    """Perform inference on batch input"""
    try:
        if CHAMPION_MODEL is None:
            return jsonify({
                'status': 'error',
                'message': 'No champion model set. Please set a champion model first.'
            }), 400
        
        data = request.json
        
        if 'instances' not in data:
            return jsonify({
                'status': 'error',
                'message': 'Input must contain "instances" key with list of feature dictionaries'
            }), 400
        
        instances = data['instances']
        if not isinstance(instances, list):
            return jsonify({
                'status': 'error',
                'message': 'Instances must be a list'
            }), 400
        
        predictions = []
        required_features = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width']
        
        for i, instance in enumerate(instances):
            # Validate each instance
            for feature in required_features:
                if feature not in instance:
                    return jsonify({
                        'status': 'error',
                        'message': f'Missing feature {feature} in instance {i}'
                    }), 400
            
            # Prepare input
            input_features = np.array([[
                instance['sepal_length'],
                instance['sepal_width'],
                instance['petal_length'],
                instance['petal_width']
            ]])
            
            # Make prediction
            prediction = CHAMPION_MODEL.predict(input_features)[0]
            
            # Get probabilities if available
            try:
                probabilities = CHAMPION_MODEL.predict_proba(input_features)[0]
                prob_dict = {IRIS_CLASS_NAMES[j]: float(probabilities[j]) for j in range(len(IRIS_CLASS_NAMES))}
            except:
                prob_dict = None
            
            predictions.append({
                'class': IRIS_CLASS_NAMES[prediction],
                'class_index': int(prediction),
                'probabilities': prob_dict,
                'input_features': instance
            })
        
        response = {
            'status': 'success',
            'predictions': predictions,
            'model_used': CHAMPION_MODEL_NAME,
            'batch_size': len(predictions)
        }
        
        return jsonify(response)
        
    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': f'Batch prediction error: {str(e)}'
        }), 500

@app.route('/champion_info', methods=['GET'])
def champion_info():
    """Get information about current champion model"""
    if CHAMPION_MODEL is None:
        return jsonify({
            'status': 'error',
            'message': 'No champion model set'
        }), 400
    
    return jsonify({
        'status': 'success',
        'champion_model': CHAMPION_MODEL_NAME,
        'model_type': type(CHAMPION_MODEL).__name__,
        'feature_names': IRIS_FEATURE_NAMES,
        'class_names': IRIS_CLASS_NAMES
    })

@app.route('/available_models', methods=['GET'])
def available_models():
    """List all available registered models"""
    try:
        registered_models = client.search_registered_models()
        models_info = []
        
        for model in registered_models:
            latest_version = client.get_latest_versions(model.name, stages=["None", "Production", "Staging"])
            versions = [v.version for v in latest_version]
            
            models_info.append({
                'name': model.name,
                'description': model.description,
                'available_versions': versions
            })
        
        return jsonify({
            'status': 'success',
            'available_models': models_info
        })
        
    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': f'Error fetching models: {str(e)}'
        }), 500

if __name__ == '__main__':
    print("🚀 Starting MLflow Flask Champion Model App...")
    
    # Setup initial models
    setup_initial_models()
    
    # Load default champion model
    load_champion_model("Champion_RandomForest", "latest")
    
    print("\n" + "="*60)
    print("FLASK APP ENDPOINTS:")
    print("="*60)
    print("🏠 Home (Web Interface):     http://localhost:5000/")
    print("🏆 Set Champion Model:       POST /set_champion")
    print("🔮 Single Prediction:        POST /predict")
    print("📊 Batch Prediction:         POST /predict_batch")
    print("ℹ️  Champion Info:            GET /champion_info")
    print("📋 Available Models:         GET /available_models")
    print("="*60)
    print("\n🌟 Open http://localhost:5000 in your browser to use the web interface!")
    
    app.run(debug=True, host='0.0.0.0', port=5000)