from flask import Flask, request, jsonify
from kafka_producer import IrisDataProducer
import pandas as pd

app = Flask(__name__)
producer = IrisDataProducer()
IRIS_HEADERS = {'sepal_length', 'sepal_width', 'petal_length', 'petal_width', 'species'}

def validate_iris_record(record):
    """Check if a record contains all required Iris fields"""
    return IRIS_HEADERS.issubset(record.keys())

@app.route('/iris', methods=['POST'])
def process_iris():
    if not request.is_json:
        return jsonify({"error": "Request must be JSON"}), 400
    
    data = request.get_json()
    
    records = data if isinstance(data, list) else [data]
    
    for record in records:
        if not validate_iris_record(record):
            return jsonify({
                "error": "Invalid Iris data structure",
                "expected_fields": list(IRIS_HEADERS),
                "received_record": record
            }), 400
    
    success_count = 0
    for record in records:
        if producer.send_iris_record(record):
            success_count += 1
        else:
            return jsonify({"error": "Failed to send data to Kafka"}), 500
    
    return jsonify({
        "message": f"Successfully processed {success_count} records",
        "records_sent": success_count
    })

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)