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
        # 1. Temperature Data
        temperature = round(random.uniform(2.0, 6.0), 2)  # Blood cold chain temp
        
        # 2. Environmental Conditions
        humidity = random.randint(30, 70)
        # Simulate a small chance of a shock event
        shock = "None" if random.random() > 0.9 else "Minor"
        ambient_temp = round(random.uniform(15.0, 30.0), 2)

        # 3. Inventory Data
        blood_types = ["O+", "O-", "A+", "A-", "B+", "B-", "AB+", "AB-"]
        inventory = {
            "units_stored": random.randint(5, 50),
            "blood_type": random.choice(blood_types),
            "expiry_date": f"2025-09-{random.randint(1, 30)}"
        }

        # 4. Pod Health
        pod_health = {
            "connectivity": "online" if random.random() > 0.05 else "intermittent",
            "battery_level": random.randint(10, 100),
            "sensor_status": "OK" if random.random() > 0.1 else "Faulty"
        }

        # 5. Alerts & Events
        alerts = []
        if temperature < 2.5 or temperature > 5.5:
            alerts.append("Temperature Deviation")
        if pod_health["connectivity"] == "intermittent":
            alerts.append("Connectivity Issue")
        if pod_health["battery_level"] < 20:
            alerts.append("Low Battery")

        # 6. Location & Tracking (simplified for simulation)
        location = {
            "latitude": round(random.uniform(12.9, 13.1), 4),
            "longitude": round(random.uniform(77.5, 77.7), 4)
        }

        # 7. Safety and Compliance Metrics (simulated expiry alert)
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
