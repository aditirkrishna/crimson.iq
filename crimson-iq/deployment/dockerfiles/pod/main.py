# main.py for a generic Pod service
import os, time, json, random
import paho.mqtt.client as mqtt

MQTT_BROKER_HOST = "mosquitto"
MQTT_PORT = 1883
TOPIC_BASE = "pods"

POD_ID = os.environ.get("ID", "default_pod")
CLIENT_ID = f"pod_{POD_ID}"

RUNTIME_SECS = int(os.getenv("RUNTIME_SECS", "180"))

print(f"Starting Pod {POD_ID}, runtime {RUNTIME_SECS}s")

client = mqtt.Client(client_id=CLIENT_ID, clean_session=True)

def on_connect(client, userdata, flags, rc):
    if rc == 0:
        print(f"Pod {POD_ID} connected.")
    else:
        print(f"Failed to connect, rc={rc}")

client.on_connect = on_connect
client.connect(MQTT_BROKER_HOST, MQTT_PORT, 60)
client.loop_start()

def simulate_pod_data():
    start = time.time()
    while time.time() - start < RUNTIME_SECS:
        temperature = round(random.uniform(2.0, 6.0), 2)
        humidity = random.randint(30, 70)
        shock = "None" if random.random() > 0.9 else "Minor"
        ambient_temp = round(random.uniform(15.0, 30.0), 2)

        blood_types = ["O+", "O-", "A+", "A-", "B+", "B-", "AB+", "AB-"]
        inventory = {
            "units_stored": random.randint(5, 50),
            "blood_type": random.choice(blood_types),
            "expiry_date": f"2025-09-{random.randint(1, 30)}"
        }

        pod_health = {
            "connectivity": "online" if random.random() > 0.05 else "intermittent",
            "battery_level": random.randint(10, 100),
            "sensor_status": "OK" if random.random() > 0.1 else "Faulty"
        }

        alerts = []
        if temperature < 2.5 or temperature > 5.5:
            alerts.append("Temperature Deviation")
        if pod_health["connectivity"] == "intermittent":
            alerts.append("Connectivity Issue")
        if pod_health["battery_level"] < 20:
            alerts.append("Low Battery")

        location = {
            "latitude": round(random.uniform(12.9, 13.1), 4),
            "longitude": round(random.uniform(77.5, 77.7), 4)
        }

        compliance = {
            "temp_compliance": "OK",
            "expiry_alert": "None" if random.random() > 0.95 else "Critical"
        }

        payload = {
            "pod_id": POD_ID,
            "timestamp": time.time(),
            "temperature_data": {
                "temperature": temperature,
                "ambient_temp": ambient_temp,
                "sensor_id": f"sensor-{POD_ID}-temp"
            },
            "environmental_data": {
                "humidity": humidity,
                "shock": shock
            },
            "inventory_data": inventory,
            "health_data": pod_health,
            "alerts": alerts,
            "location": location,
            "compliance_data": compliance
        }

        topic = f"{TOPIC_BASE}/{POD_ID}/status"
        result = client.publish(topic, json.dumps(payload), qos=1)
        if result[0] == mqtt.MQTT_ERR_SUCCESS:
            print(f"Published to {topic}: {payload}")
        time.sleep(random.randint(5, 15))

    print(f"Pod {POD_ID} finished runtime.")
    client.loop_stop()
    client.disconnect()

if __name__ == "__main__":
    simulate_pod_data()
