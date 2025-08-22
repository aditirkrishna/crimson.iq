# Docker Configuration

This directory contains Dockerfiles for all services:
- inventory-service.Dockerfile
- reallocation-service.Dockerfile
- fl-orchestrator.Dockerfile
- frontend.Dockerfile
- edge-device.Dockerfile

# Crimson.IQ Simulation Platform 🚑🩸

Crimson.IQ is a containerized simulation platform for modeling a **smart blood supply chain network**.  
It uses **Docker**, **MQTT**, and **MongoDB** to simulate interactions between **Pods**, **Hospitals**, and **Blood Banks** in real-time.  

---

## 📦 System Architecture

### 🔹 Core Components
- **Pods** (`pod/`)  
  Simulate smart storage units carrying blood samples with sensors for temperature, humidity, battery, compliance, etc.  

- **Hospitals** (`hospital/`)  
  Simulate hospital demand and supply requests for blood units.  

- **Blood Banks** (`bloodbank/`)  
  Simulate storage and dispatch operations of blood inventory.  

- **Collector** (`collector/`)  
  Central data logger that subscribes to all MQTT topics and:
  - Inserts data into **MongoDB** (for query, persistence, and audits).  
  - Saves data into two dump files inside `/data/`:  
    - `events_raw.ndjson` → Mongo-style, full documents.  
    - `events_ml.ndjson` → Flattened, ML-ready dataset.  

- **GUI/Dashboard** (`gui/`)  
  A lightweight **Tailwind + Nginx** frontend that shows live updates from the MQTT broker.  

- **Mosquitto Broker** (`mosquitto`)  
  The centralized MQTT broker enabling real-time communication between all services.  

- **MongoDB** (`mongodb`)  
  Provides reliable, scalable, and fast database storage for all simulation events.  

---

## 🔗 How It Works
1. Each **pod**, **hospital**, and **blood bank** runs as an independent Docker container.  
2. They **publish their data** (status, inventory, temperature, demand, etc.) to the **Mosquitto MQTT broker**.  
3. The **collector service** listens to all topics, enriches the messages with metadata, and:  
   - Stores them into MongoDB.  
   - Dumps them to flat NDJSON files for later ML/analytics.  
4. The **GUI** connects via WebSockets to the MQTT broker and visualizes the system status in real time.  

This architecture mirrors a real-world **IoT + Healthcare Data Pipeline**.  

---

## 🐳 Docker & Containerization
- Each service (`pod`, `hospital`, `bloodbank`, `collector`, `gui`) is **isolated in its own container**.  
- Communication between services happens via **Docker’s internal network** (service names like `mosquitto`, `mongodb`).  
- The setup is orchestrated using **docker-compose** for easy multi-service deployment.  

Benefits:
- ✅ **Scalability**: Add more pods/hospitals with a single config line.  
- ✅ **Reproducibility**: Same setup runs on any machine with Docker.  
- ✅ **Isolation**: Bugs in one service don’t crash others.  
- ✅ **Portability**: Works across Linux, Mac, Windows with no extra setup.  

---

## 📂 Project Structure
crimson-iq/ <br>
├── docker-compose.yml # Orchestration config<br>
├── conf/<br>
│ └── mosquitto.conf # MQTT broker config<br>
├── data/ # All dataset exports<br>
│ ├── events_raw.ndjson # Mongo-style dump<br>
│ ├── events_ml.ndjson # Flattened ML dataset<br>
│ └── mongo/ # MongoDB volume persistence<br>
├── gui/<br>
│ ├── index.html # Frontend dashboard<br>
│ ├── nginx.conf # Webserver config<br>
│ └── Dockerfile<br>
├── collector/<br>
│ ├── main.py # Collector service<br>
│ ├── requirements.txt<br>
│ └── Dockerfile<br>
├── pod/<br>
│ ├── main.py # Pod simulator<br>
│ ├── requirements.txt<br>
│ └── Dockerfile<br>
├── hospital/<br>
│ ├── main.py # Hospital simulator<br>
│ ├── requirements.txt<br>
│ └── Dockerfile<br>
├── bloodbank/<br>
│ ├── main.py # Blood bank simulator<br>
│ ├── requirements.txt<br>
│ └── Dockerfile<br>
└── README.md<br>


---

## 🚀 Running the System

Make sure **Docker** & **Docker Compose** are installed.

```bash
# Clone the repository
git clone https://github.com/your-username/crimson-iq.git
cd crimson-iq

# Start the simulation
docker-compose up --build


The system will spin up:

Mosquitto broker (1883 for MQTT, 9001 for WebSockets)

MongoDB database (27017)

GUI at http://localhost:8080

Collector and all publishers (pods, hospitals, bloodbanks)

📊 Data Storage & Export

MongoDB
Events are stored with _id, ISODate, and raw payloads — ideal for queries, audits, and dashboards.

NDJSON Dumps

events_raw.ndjson → full Mongo-style records (audit trail).

events_ml.ndjson → flattened & ML-ready (ISO timestamps, strings/numbers/booleans only).

⚙️ Environment Variables (common)

MQTT_HOST (default: mosquitto)

MQTT_PORT (default: 1883)

MQTT_TOPICS (collector; default: #)

MONGO_URI (default: mongodb://mongodb:27017)

MONGO_DB (default: crimson)

MONGO_COLLECTION (default: events)

DUMP_NDJSON_PATH (collector; default: /data/events_raw.ndjson)

DUMP_ML_NDJSON_PATH (collector; default: /data/events_ml.ndjson)

RUNTIME_SECS (shared duration for runs; e.g., 60, 120, etc.)