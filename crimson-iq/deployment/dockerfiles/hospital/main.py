# main.py for a generic Hospital service
import os, time, json, random
import paho.mqtt.client as mqtt

MQTT_BROKER_HOST = "mosquitto"
MQTT_PORT = 1883
TOPIC_BASE = "hospital"

HOSPITAL_ID = os.environ.get("ID", "default_hospital")
CLIENT_ID = f"hospital_{HOSPITAL_ID}"

RUNTIME_SECS = int(os.getenv("RUNTIME_SECS", "180"))

print(f"Starting Hospital {HOSPITAL_ID}, runtime {RUNTIME_SECS}s")

client = mqtt.Client(client_id=CLIENT_ID, clean_session=True)

def on_connect(client, userdata, flags, rc):
    if rc == 0:
        print(f"Hospital {HOSPITAL_ID} connected.")
    else:
        print(f"Failed to connect, rc={rc}")

client.on_connect = on_connect
client.connect(MQTT_BROKER_HOST, MQTT_PORT, 60)
client.loop_start()

def simulate_hospital_data():
    start = time.time()
    while time.time() - start < RUNTIME_SECS:
        patients = random.randint(10, 200)
        urgent_need = random.choice([True, False])
        demand_prediction = random.randint(0, 10)

        payload = {
            "hospital_id": HOSPITAL_ID,
            "timestamp": time.time(),
            "patients": patients,
            "urgent_need": urgent_need,
            "demand_prediction": demand_prediction
        }

        topic = f"{TOPIC_BASE}/{HOSPITAL_ID}/status"
        result = client.publish(topic, json.dumps(payload), qos=1)
        if result[0] == mqtt.MQTT_ERR_SUCCESS:
            print(f"Published to {topic}: {payload}")
        time.sleep(random.randint(5, 15))

    print(f"Hospital {HOSPITAL_ID} finished runtime.")
    client.loop_stop()
    client.disconnect()

if __name__ == "__main__":
    simulate_hospital_data()
