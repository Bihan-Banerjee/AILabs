from confluent_kafka import Consumer

config = {
    'bootstrap.servers': 'localhost:57891',
    'group.id': 'multi-channel-viewer',
    'auto.offset.reset': 'earliest'
}

consumer = Consumer(config)

# You can add/remove topics here
topics = ['news', 'sports']
consumer.subscribe(topics)

print(f"[Consumer] Subscribed to: {', '.join(topics)}")

try:
    while True:
        msg = consumer.poll(1.0)
        if msg is None:
            continue
        if msg.error():
            print(f"Error: {msg.error()}")
        else:
            print(f"[{msg.topic()}] {msg.value().decode('utf-8')}")
except KeyboardInterrupt:
    print("Stopped by user")
finally:
    consumer.close()