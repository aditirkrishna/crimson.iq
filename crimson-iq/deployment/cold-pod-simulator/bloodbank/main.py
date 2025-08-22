# main.py for a generic Bloodbank service
import os, time, json, random
import paho.mqtt.client as mqtt

# MQTT Broker settings
MQTT_BROKER_HOST = "mosquitto"
MQTT_PORT = 1883
TOPIC_BASE = "bloodbank"

# Unique ID
BANK_ID = os.environ.get("ID", "default_bank")
CLIENT_ID = f"bloodbank_{BANK_ID}"

# Runtime (default 180s = 3 mins)
RUNTIME_SECS = int(os.getenv("RUNTIME_SECS", "180"))

print(f"Starting Bloodbank {BANK_ID}, runtime {RUNTIME_SECS}s")

client = mqtt.Client(client_id=CLIENT_ID, clean_session=True)

def on_connect(client, userdata, flags, rc):
    if rc == 0:
        print(f"Bloodbank {BANK_ID} connected.")
    else:
        print(f"Failed to connect, rc={rc}")

client.on_connect = on_connect
client.connect(MQTT_BROKER_HOST, MQTT_PORT, 60)
client.loop_start()

def simulate_bloodbank_data():
    start = time.time()
    while time.time() - start < RUNTIME_SECS:
        inventory_level = random.randint(50, 200)
        temperature = round(random.uniform(2.0, 6.0), 2)
        demand_prediction = random.randint(0, 10)

        payload = {
            "bank_id": BANK_ID,
            "timestamp": time.time(),
            "inventory": inventory_level,
            "temperature": temperature,
            "demand_prediction": demand_prediction
        }

        topic = f"{TOPIC_BASE}/{BANK_ID}/status"
        result = client.publish(topic, json.dumps(payload), qos=1)
        if result[0] == mqtt.MQTT_ERR_SUCCESS:
            print(f"Published to {topic}: {payload}")
        time.sleep(random.randint(5, 15))

    print(f"Bloodbank {BANK_ID} finished runtime.")
    client.loop_stop()
    client.disconnect()

if __name__ == "__main__":
    simulate_bloodbank_data()
