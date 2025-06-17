from confluent_kafka import Producer
import json

class IrisDataProducer:
    def __init__(self, bootstrap_servers='localhost:9092', topic='iris_data'):
        self.producer = Producer({'bootstrap.servers': bootstrap_servers})
        self.topic = topic

    def delivery_report(self, err, msg):
        if err:
            print(f'Message delivery failed: {err}')
        else:
            print(f'Message delivered to {msg.topic()}')

    def send_iris_record(self, record):
        try:
            self.producer.produce(
                topic=self.topic,
                value=json.dumps(record).encode('utf-8'),
                callback=self.delivery_report
            )
            self.producer.flush()
            return True
        except Exception as e:
            print(f"Failed to send message: {e}")
            return False