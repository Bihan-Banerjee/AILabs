from confluent_kafka import Producer

producer = Producer({'bootstrap.servers': 'localhost:57891'})
topic = 'sports'

messages = [
    'Sports: Team A wins the championship!',
    ' Sports: New basketball record set.'
]

for msg in messages:
    producer.produce(topic, value=msg)
    print(f"[Producer - sports] Sent: {msg}")
    producer.poll(0)

producer.flush()
from confluent_kafka import Producer

producer = Producer({'bootstrap.servers': 'localhost:57891'})
topic = 'sports'

messages = [
    'Sports: Team A wins the championship!',
    ' Sports: New basketball record set.'
]

for msg in messages:
    producer.produce(topic, value=msg)
    print(f"[Producer - sports] Sent: {msg}")
    producer.poll(0)

producer.flush()