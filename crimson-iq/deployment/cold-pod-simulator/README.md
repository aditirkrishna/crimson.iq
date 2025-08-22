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
