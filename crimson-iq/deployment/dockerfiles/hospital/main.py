# main.py for a generic Pod service
# Save this as `main.py` inside the `pod/` directory

import os
import time
import json
import random
import paho.mqtt.client as mqtt

# MQTT Broker settings
MQTT_BROKER_HOST = "mosquitto" # Docker service name
MQTT_PORT = 1883
TOPIC_BASE = "pods"

# Get the pod's unique ID from the environment variable
POD_ID = os.environ.get("ID", "default_pod")
CLIENT_ID = f"pod_{POD_ID}"

print(f"Starting Pod with ID: {POD_ID}")

# MQTT client setup
client = mqtt.Client(client_id=CLIENT_ID, clean_session=True)

def on_connect(client, userdata, flags, rc):
    """Callback function for when the client connects to the MQTT broker."""
    if rc == 0:
        print(f"Pod {POD_ID} connected to MQTT broker successfully.")
        # Subscribe to any command topics if needed
        client.subscribe(f"{TOPIC_BASE}/{POD_ID}/commands")
    else:
        print(f"Failed to connect, return code {rc}\n")

def on_message(client, userdata, msg):
    """Callback function for when a message is received from the broker."""
    print(f"Message received on topic {msg.topic}: {msg.payload.decode()}")

client.on_connect = on_connect
client.on_message = on_message

# Connect to the broker
try:
    client.connect(MQTT_BROKER_HOST, MQTT_PORT, 60)
except Exception as e:
    print(f"Connection error: {e}")
    exit(1)

# Start the MQTT loop in a non-blocking way
client.loop_start()

# Main simulation loop
def simulate_pod_data():
    """Simulates a pod's data and publishes it to the broker."""
    while True:
        # Simulate inventory and cold chain data
        inventory_level = random.randint(50, 200)
        temperature = round(random.uniform(2.0, 6.0), 2)  # Blood cold chain temp
        
        # Simple AI simulation: "Predict" demand
        demand_prediction = random.randint(0, 10)
        
        payload = {
            "pod_id": POD_ID,
            "timestamp": time.time(),
            "inventory": inventory_level,
            "temperature": temperature,
            "demand_prediction": demand_prediction
        }
        
        # Publish the data
        topic = f"{TOPIC_BASE}/{POD_ID}/status"
        result = client.publish(topic, json.dumps(payload), qos=1)
        
        status = result[0]
        if status == mqtt.MQTT_ERR_SUCCESS:
            print(f"Published to {topic}: {payload}")
        else:
            print(f"Failed to publish to {topic}")
        
        # Wait for a bit before the next update
        time.sleep(random.randint(5, 15))

if __name__ == "__main__":
    try:
        simulate_pod_data()
    except KeyboardInterrupt:
        print("Simulation stopped.")
        client.loop_stop()
        client.disconnect()