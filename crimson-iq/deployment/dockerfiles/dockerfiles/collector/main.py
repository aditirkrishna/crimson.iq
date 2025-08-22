import os, json, signal, sys, threading
from datetime import datetime, timezone
import paho.mqtt.client as mqtt
from pymongo import MongoClient, ASCENDING, DESCENDING
from pymongo.errors import PyMongoError

# --- Env ---
MQTT_HOST = os.getenv("MQTT_HOST", "mosquitto")
MQTT_PORT = int(os.getenv("MQTT_PORT", "1883"))
MQTT_TOPICS = os.getenv("MQTT_TOPICS", "#")  # subscribe to everything

MONGO_URI = os.getenv("MONGO_URI", "mongodb://mongodb:27017")
MONGO_DB = os.getenv("MONGO_DB", "crimson")
MONGO_COLLECTION = os.getenv("MONGO_COLLECTION", "events")

DUMP_PATH = os.getenv("DUMP_NDJSON_PATH", "/data/events.ndjson")
RUN_DURATION = int(os.getenv("RUNTIME_SECS", "180"))  # default: 3 minutes

# --- Mongo ---
mongo_client = MongoClient(MONGO_URI)
db = mongo_client[MONGO_DB]
col = db[MONGO_COLLECTION]

# Helpful indexes for query speed
col.create_index([("ts", DESCENDING)])
col.create_index([("topic", ASCENDING)])
col.create_index([("source", ASCENDING), ("entity_id", ASCENDING)])

# --- File dump (newline-delimited JSON) ---
dump_file = open(DUMP_PATH, "a", buffering=1)

is_shutting_down = False

def close_and_exit(*_):
    global is_shutting_down
    is_shutting_down = True
    print("Collector shutting down...", flush=True)
    try:
        dump_file.flush()
        dump_file.close()
    except Exception:
        pass
    try:
        mqtt_client.loop_stop()
    except Exception:
        pass
    try:
        mongo_client.close()
    except Exception:
        pass
    sys.exit(0)

signal.signal(signal.SIGINT, close_and_exit)
signal.signal(signal.SIGTERM, close_and_exit)

# Auto-shutdown after RUN_DURATION seconds
def shutdown_timer():
    print(f"Collector will run for {RUN_DURATION} seconds then exit.", flush=True)
    threading.Timer(RUN_DURATION, close_and_exit).start()

def parse_topic(topic: str):
    parts = topic.split("/")
    source = parts[0] if len(parts) > 0 else None
    entity_id = parts[1] if len(parts) > 1 else None
    event_type = parts[2] if len(parts) > 2 else None
    return source, entity_id, event_type

def on_connect(client, userdata, flags, rc):
    print("MQTT connected with rc:", rc, flush=True)
    client.subscribe(MQTT_TOPICS)
    print("Subscribed to:", MQTT_TOPICS, flush=True)

def on_message(client, userdata, msg):
    global is_shutting_down
    if is_shutting_down:
        return  # ignore late packets during shutdown

    try:
        payload = json.loads(msg.payload.decode("utf-8"))
    except Exception:
        payload = {"raw": msg.payload.decode("utf-8", errors="ignore")}

    source, entity_id, event_type = parse_topic(msg.topic)
    doc = {
        "ts": datetime.now(timezone.utc),  # Mongo stores as Date
        "topic": msg.topic,
        "source": source,
        "entity_id": entity_id,
        "event_type": event_type,
        "payload": payload
    }

    # 1) Insert into Mongo
    try:
        col.insert_one(doc)
    except PyMongoError as e:
        print("Mongo insert failed:", e, flush=True)

    # 2) Append to single NDJSON file (ISO timestamp for portability)
    dump_doc = dict(doc)
    dump_doc["ts"] = doc["ts"].isoformat()
    dump_doc.pop("_id", None)  # remove Mongo's ObjectId
    try:
        dump_file.write(json.dumps(dump_doc) + "\n")
    except Exception as e:
        print("File write failed:", e, flush=True)

# --- MQTT ---
mqtt_client = mqtt.Client()
mqtt_client.on_connect = on_connect
mqtt_client.on_message = on_message

mqtt_client.connect(MQTT_HOST, MQTT_PORT, 60)

# Start timer + loop
shutdown_timer()
mqtt_client.loop_forever()
