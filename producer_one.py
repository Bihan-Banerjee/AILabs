from confluent_kafka import Producer

producer = Producer({'bootstrap.servers': 'localhost:57891'})
topic = 'news'

messages = [
    ' News: AI beats human in chess!',
    ' News: Global warming rising fast.'
]

for msg in messages:
    producer.produce(topic, value=msg)
    print(f"[Producer - news] Sent: {msg}")
    producer.poll(0)

producer.flush()