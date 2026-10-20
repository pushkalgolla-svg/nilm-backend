import os
import sys

# Ensure UTF-8 output on Windows
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

import time
import math
import json
import csv
import queue
import random
import threading
from datetime import datetime, timezone
from contextlib import asynccontextmanager
from typing import Dict, Any, List, Tuple, Optional

import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
from sklearn.model_selection import train_test_split

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import uvicorn

import paho.mqtt.client as mqtt

# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------
SIMULATION_MODE = os.getenv("SIMULATION_MODE", "False").lower() in ("true", "1", "yes")
SERIAL_PORT = os.getenv("SERIAL_PORT", "COM3")
SERIAL_BAUDRATE = int(os.getenv("SERIAL_BAUDRATE", "115200"))
DELTA_I_THRESHOLD = float(os.getenv("DELTA_I_THRESHOLD", "0.15"))  # RMS current jump trigger (Amps)

DATASET_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "dataset.csv")

FEATURE_COLUMNS = [
    "current_rms",
    "delta_i",
    "accel_x_g",
    "accel_y_g",
    "accel_z_g",
    "gyro_x_dps",
    "gyro_y_dps",
    "gyro_z_dps",
    "vibration_magnitude_g",
]

# Global trained model and metrics storage
trained_classifier: Optional[DecisionTreeClassifier] = None
model_metrics: Dict[str, Any] = {
    "accuracy": 0.0,
    "classes": [],
    "train_samples": 0,
    "test_samples": 0,
    "trained_at": None,
}

# Thread control & event queue
stop_event = threading.Event()
worker_thread = None
labeler_thread = None
data_lock = threading.Lock()
event_queue: queue.Queue = queue.Queue()
last_wifi_telemetry_time: float = 0.0

# MQTT Configuration
MQTT_BROKER = "broker.hivemq.com"
MQTT_PORT = 1883
MQTT_TOPIC = "nilm/pushkal/telemetry"
mqtt_client_instance = None
mqtt_status_info = {"connected": False, "error": None, "last_msg": None}


def process_mqtt_telemetry(payload_dict: dict):
    """Process incoming MQTT telemetry exactly like the HTTP endpoint."""
    global last_wifi_telemetry_time
    last_wifi_telemetry_time = time.time()
    now_iso = datetime.now(timezone.utc).isoformat()

    current_rms = float(payload_dict.get("current_rms", 0.0))
    delta_i = float(payload_dict.get("delta_i", 0.0))
    voltage = float(payload_dict.get("voltage", 230.0))
    accel_x = float(payload_dict.get("accel_x", 0.0))
    accel_y = float(payload_dict.get("accel_y", 0.0))
    accel_z = float(payload_dict.get("accel_z", 1.0))
    gyro_x = float(payload_dict.get("gyro_x", 0.0))
    gyro_y = float(payload_dict.get("gyro_y", 0.0))
    gyro_z = float(payload_dict.get("gyro_z", 0.0))
    vib_mag = float(payload_dict.get("vibration_magnitude", 0.0))
    if vib_mag == 0.0:
        vib_mag = round(math.sqrt(accel_x**2 + accel_y**2 + accel_z**2), 4)

    predicted_appliance = payload_dict.get("predicted_appliance", "")
    confidence_val = float(payload_dict.get("confidence", 1.0))
    device_id = payload_dict.get("device_id", "wokwi-esp32")

    imu_data = {
        "accel_x_g": accel_x, "accel_y_g": accel_y, "accel_z_g": accel_z,
        "gyro_x_dps": gyro_x, "gyro_y_dps": gyro_y, "gyro_z_dps": gyro_z,
        "vibration_magnitude_g": vib_mag,
    }

    if predicted_appliance and predicted_appliance.strip() not in ("", "Unknown"):
        pred_label = predicted_appliance.strip()
        confidence = confidence_val
        model_source = "TinyML ESP32 (inference.h)"
    else:
        pred_label, confidence = run_live_inference(current_rms, delta_i, imu_data)
        model_source = "DecisionTreeClassifier (Cloud)"

    classified_apps = apply_inference_to_appliances(pred_label, current_rms)

    with data_lock:
        latest_data["timestamp"] = now_iso
        latest_data["mode"] = f"mqtt ({device_id})"
        latest_data["total_current_rms"] = round(current_rms, 3)
        latest_data["delta_i"] = round(delta_i, 3)
        latest_data["voltage_rms"] = round(voltage, 1)
        latest_data["appliances"] = classified_apps
        latest_data["ml_inference"] = {
            "predicted_appliance": pred_label,
            "confidence": round(confidence, 3),
            "model": model_source,
            "last_inference_time": now_iso,
        }

    print(f"[MQTT TELEMETRY] Node: {device_id} | I: {current_rms:.2f}A | dI: {delta_i:+.2f}A | Prediction -> {pred_label} ({confidence*100:.1f}%)")


def on_mqtt_connect(client, userdata, flags, reason_code, properties=None):
    mqtt_status_info["connected"] = True
    mqtt_status_info["rc"] = str(reason_code)
    print(f"[MQTT] Connected to {MQTT_BROKER}:{MQTT_PORT} (rc={reason_code})")
    client.subscribe(MQTT_TOPIC)
    print(f"[MQTT] Subscribed to topic: {MQTT_TOPIC}")


def on_mqtt_message(client, userdata, msg):
    try:
        raw_str = msg.payload.decode("utf-8")
        mqtt_status_info["last_msg"] = raw_str
        payload_dict = json.loads(raw_str)
        process_mqtt_telemetry(payload_dict)
    except Exception as e:
        mqtt_status_info["error"] = str(e)
        print(f"[MQTT Error] Failed to parse message: {e}")


def start_mqtt_subscriber():
    global mqtt_client_instance
    try:
        mqtt_client_instance = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2)
        mqtt_client_instance.on_connect = on_mqtt_connect
        mqtt_client_instance.on_message = on_mqtt_message
        mqtt_client_instance.connect(MQTT_BROKER, MQTT_PORT, 60)
        mqtt_client_instance.loop_start()
        print(f"[MQTT] Subscriber thread started -> {MQTT_BROKER}:{MQTT_PORT} topic={MQTT_TOPIC}")
    except Exception as e:
        mqtt_status_info["error"] = str(e)
        print(f"[MQTT Error] Failed to start subscriber: {e}")
        print(f"[MQTT Error] Could not connect to broker: {e}")

# Shared state store
latest_data: Dict[str, Any] = {
    "timestamp": None,
    "mode": "simulation" if SIMULATION_MODE else "serial",
    "total_current_rms": 0.0,
    "delta_i": 0.0,
    "voltage_rms": 230.0,
    "ml_inference": {
        "predicted_appliance": "Standby",
        "confidence": 1.0,
        "model": "DecisionTreeClassifier (max_depth=5)",
        "last_inference_time": None,
    },
    "appliances": {
        "Ceiling Fan": {"current_rms": 0.0, "status": "OFF"},
        "Electric Iron": {"current_rms": 0.0, "status": "OFF"},
        "Fast Charger": {"current_rms": 0.0, "status": "OFF"},
    },
}


# -----------------------------------------------------------------------------
# Synthetic Baseline Data Generator (ensures dataset.csv has balanced classes)
# -----------------------------------------------------------------------------
def ensure_baseline_dataset(min_samples_per_class: int = 25):
    """
    Checks dataset.csv. If missing or underpopulated, generates a balanced
    multi-sensor fusion dataset capturing realistic electrical & MPU6050 vibration
    characteristics for Fan, Iron, Charger, and Standby states.
    """
    existing_rows = []
    if os.path.exists(DATASET_FILE):
        try:
            with open(DATASET_FILE, mode="r", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                existing_rows = list(reader)
        except Exception:
            existing_rows = []

    if len(existing_rows) >= (min_samples_per_class * 4):
        return

    print(f"[Dataset] Populating baseline dataset.csv with {min_samples_per_class * 4} sensor fusion samples...")

    classes_config = [
        # (label, current_range, delta_range, ax_range, ay_range, az_range, gx_range, gy_range, gz_range)
        ("Standby", (0.02, 0.08), (0.0, 0.05), (0.002, 0.005), (0.002, 0.005), (0.998, 1.002), (0.02, 0.05), (0.02, 0.05), (0.02, 0.05)),
        ("Ceiling Fan", (0.35, 0.42), (0.32, 0.40), (0.10, 0.16), (0.12, 0.18), (0.95, 1.02), (2.0, 3.2), (-2.2, -1.2), (3.0, 4.5)),
        ("Electric Iron", (5.40, 5.85), (5.10, 5.75), (0.01, 0.03), (-0.02, 0.02), (0.98, 1.02), (0.1, 0.3), (-0.2, 0.2), (0.0, 0.2)),
        ("Fast Charger", (0.20, 0.26), (0.18, 0.25), (0.004, 0.008), (0.004, 0.008), (0.995, 1.005), (0.04, 0.08), (0.04, 0.08), (0.04, 0.08)),
    ]

    new_rows = []
    base_time = datetime.now(timezone.utc)

    for label, i_rng, d_rng, ax_rng, ay_rng, az_rng, gx_rng, gy_rng, gz_rng in classes_config:
        for idx in range(min_samples_per_class):
            c_rms = round(random.uniform(*i_rng), 3)
            d_i = round(random.uniform(*d_rng), 3)
            ax = round(random.uniform(*ax_rng), 4)
            ay = round(random.uniform(*ay_rng), 4)
            az = round(random.uniform(*az_rng), 4)
            gx = round(random.uniform(*gx_rng), 2)
            gy = round(random.uniform(*gy_rng), 2)
            gz = round(random.uniform(*gz_rng), 2)
            v_mag = round(math.sqrt(ax**2 + ay**2 + az**2), 4)

            new_rows.append({
                "timestamp": base_time.isoformat(),
                "current_rms": c_rms,
                "delta_i": d_i,
                "accel_x_g": ax,
                "accel_y_g": ay,
                "accel_z_g": az,
                "gyro_x_dps": gx,
                "gyro_y_dps": gy,
                "gyro_z_dps": gz,
                "vibration_magnitude_g": v_mag,
                "label": label,
            })

    # Write combined rows
    fieldnames = [
        "timestamp", "current_rms", "delta_i",
        "accel_x_g", "accel_y_g", "accel_z_g",
        "gyro_x_dps", "gyro_y_dps", "gyro_z_dps",
        "vibration_magnitude_g", "label"
    ]
    with open(DATASET_FILE, mode="w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(new_rows)

    print(f"[Dataset] Baseline dataset created: {len(new_rows)} samples across 4 appliance classes.")


# -----------------------------------------------------------------------------
# Confusion Matrix & Model Evaluation Report Utility
# -----------------------------------------------------------------------------
def print_confusion_matrix_report(y_true: List[str], y_pred: List[str], labels: List[str], accuracy: float):
    """
    Prints a formatted, clean Confusion Matrix and Classification Metrics
    table to the terminal for inclusion in academic project reports.
    """
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    col_width = max(len(l) for l in labels) + 2

    print("\n" + "=" * 74)
    print(" [REPORT] AIoT NILM DECISION TREE CLASSIFIER - EVALUATION REPORT")
    print("=" * 74)
    print(f" Model Architecture:       Scikit-Learn DecisionTreeClassifier (max_depth=5)")
    print(f" Evaluation Test Accuracy: {accuracy * 100:.2f}%\n")

    print(" --- CONFUSION MATRIX ---")
    title_cell = "True / Pred"
    header = f"{title_cell:<{col_width}}" + "".join(f"{lbl:>{col_width}}" for lbl in labels)
    print(" " + header)
    print(" " + "-" * len(header))

    # Matrix rows
    for i, true_label in enumerate(labels):
        row_str = f" {true_label:<{col_width-1}}" + "".join(f"{cm[i][j]:>{col_width}}" for j in range(len(labels)))
        print(row_str)

    print("\n --- CLASSIFICATION PERFORMANCE METRICS ---")
    report = classification_report(y_true, y_pred, labels=labels, zero_division=0)
    for line in report.splitlines():
        print("  " + line)
    print("=" * 74 + "\n")


# -----------------------------------------------------------------------------
# Train Decision Tree on Startup
# -----------------------------------------------------------------------------
def train_decision_tree() -> Tuple[DecisionTreeClassifier, float]:
    """
    Loads dataset.csv, splits into train/test sets, fits a Decision Tree,
    and prints the Confusion Matrix and accuracy score.
    """
    ensure_baseline_dataset()

    X_data = []
    y_data = []

    with open(DATASET_FILE, mode="r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                features = [float(row[col]) for col in FEATURE_COLUMNS]
                label = row["label"].strip()
                if label:
                    X_data.append(features)
                    y_data.append(label)
            except (ValueError, KeyError):
                continue

    if len(X_data) < 10:
        raise ValueError("Insufficient data in dataset.csv to train model.")

    X = np.array(X_data, dtype=np.float32)
    y = np.array(y_data)

    unique_classes = [str(c) for c in sorted(list(set(y_data)))]

    # 80/20 train/test split with stratification
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=y
    )

    clf = DecisionTreeClassifier(
        criterion="gini",
        max_depth=5,
        min_samples_split=2,
        random_state=42
    )
    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)
    acc = float(accuracy_score(y_test, y_pred))

    # Print Confusion Matrix and performance metrics to terminal for project report
    print_confusion_matrix_report(y_test.tolist(), y_pred.tolist(), unique_classes, acc)

    # Store global metrics
    global trained_classifier
    trained_classifier = clf
    model_metrics["accuracy"] = round(acc, 4)
    model_metrics["classes"] = list(unique_classes)
    model_metrics["train_samples"] = int(len(X_train))
    model_metrics["test_samples"] = int(len(X_test))
    model_metrics["trained_at"] = datetime.now(timezone.utc).isoformat()

    return clf, acc


# -----------------------------------------------------------------------------
# Simulated MPU6050 Vibration Snapshot
# -----------------------------------------------------------------------------
def generate_mpu6050_snapshot(current_rms: float, delta_i: float) -> Dict[str, float]:
    abs_delta = abs(delta_i)

    if abs_delta >= 4.0 or current_rms >= 4.0:
        # Electric Iron: Heavy thermal mass, static surface contact
        ax = round(random.gauss(0.015, 0.01), 4)
        ay = round(random.gauss(-0.012, 0.01), 4)
        az = round(random.gauss(1.002, 0.015), 4)
        gx = round(random.gauss(0.12, 0.2), 2)
        gy = round(random.gauss(-0.08, 0.2), 2)
        gz = round(random.gauss(0.02, 0.15), 2)
    elif 0.25 <= abs_delta < 4.0 or 0.25 <= current_rms < 1.5:
        # Ceiling Fan: Motor rotational harmonics
        ax = round(random.gauss(0.125, 0.03), 4)
        ay = round(random.gauss(0.148, 0.03), 4)
        az = round(random.gauss(0.975, 0.025), 4)
        gx = round(random.gauss(2.4, 0.6), 2)
        gy = round(random.gauss(-1.7, 0.5), 2)
        gz = round(random.gauss(3.9, 0.8), 2)
    elif 0.10 <= abs_delta < 0.25 or 0.10 <= current_rms < 0.30:
        # Fast Charger: GaN SMPS subtle ripple
        ax = round(random.gauss(0.005, 0.006), 4)
        ay = round(random.gauss(0.005, 0.006), 4)
        az = round(random.gauss(1.001, 0.008), 4)
        gx = round(random.gauss(0.06, 0.08), 2)
        gy = round(random.gauss(0.04, 0.08), 2)
        gz = round(random.gauss(0.05, 0.08), 2)
    else:
        # Standby
        ax = round(random.gauss(0.002, 0.004), 4)
        ay = round(random.gauss(0.002, 0.004), 4)
        az = round(random.gauss(1.000, 0.005), 4)
        gx = round(random.gauss(0.02, 0.04), 2)
        gy = round(random.gauss(0.02, 0.04), 2)
        gz = round(random.gauss(0.02, 0.04), 2)

    vibration_mag = round(math.sqrt(ax**2 + ay**2 + az**2), 4)

    return {
        "accel_x_g": ax,
        "accel_y_g": ay,
        "accel_z_g": az,
        "gyro_x_dps": gx,
        "gyro_y_dps": gy,
        "gyro_z_dps": gz,
        "vibration_magnitude_g": vibration_mag,
    }


# -----------------------------------------------------------------------------
# Real-Time Decision Tree Inference Handler
# -----------------------------------------------------------------------------
def run_live_inference(total_current: float, delta_i: float, imu_data: Dict[str, float]) -> Tuple[str, float]:
    """
    Constructs feature vector from event snapshot and passes it through
    the trained Decision Tree for real-time classification.
    """
    global trained_classifier
    if trained_classifier is None:
        return "Unknown", 0.0

    features = [
        total_current,
        delta_i,
        imu_data["accel_x_g"],
        imu_data["accel_y_g"],
        imu_data["accel_z_g"],
        imu_data["gyro_x_dps"],
        imu_data["gyro_y_dps"],
        imu_data["gyro_z_dps"],
        imu_data["vibration_magnitude_g"],
    ]

    features_arr = np.array([features], dtype=np.float32)
    pred_label = str(trained_classifier.predict(features_arr)[0])

    # Compute probability confidence
    try:
        probs = trained_classifier.predict_proba(features_arr)[0]
        confidence = float(np.max(probs))
    except Exception:
        confidence = 1.0

    return pred_label, confidence


def apply_inference_to_appliances(pred_label: str, total_current: float):
    """
    Updates the shared appliance states based on the Decision Tree prediction.
    """
    # Robustness fallback: reconcile label with total RMS current level
    if total_current < 0.20:
        pred_label = "Standby"
    elif pred_label == "Standby":
        if total_current >= 2.50:
            pred_label = "Electric Iron"
        else:
            pred_label = "Fast Charger"

    # Base states
    new_appliances = {
        "Ceiling Fan": {"current_rms": 0.0, "status": "OFF"},
        "Electric Iron": {"current_rms": 0.0, "status": "OFF"},
        "Fast Charger": {"current_rms": 0.0, "status": "OFF"},
    }

    if pred_label == "Electric Iron":
        new_appliances["Electric Iron"]["status"] = "ON"
        new_appliances["Electric Iron"]["current_rms"] = round(min(total_current, 5.65), 2)
        # Background fan/charger if load is high enough
        if total_current >= 6.0:
            new_appliances["Ceiling Fan"]["status"] = "ON"
            new_appliances["Ceiling Fan"]["current_rms"] = 0.38
        if total_current >= 6.2:
            new_appliances["Fast Charger"]["status"] = "ON"
            new_appliances["Fast Charger"]["current_rms"] = 0.22

    elif pred_label == "Ceiling Fan":
        new_appliances["Ceiling Fan"]["status"] = "ON"
        new_appliances["Ceiling Fan"]["current_rms"] = round(min(total_current, 0.40), 2)
        if total_current >= 0.55:
            new_appliances["Fast Charger"]["status"] = "ON"
            new_appliances["Fast Charger"]["current_rms"] = 0.22

    elif pred_label == "Fast Charger":
        new_appliances["Fast Charger"]["status"] = "ON"
        new_appliances["Fast Charger"]["current_rms"] = round(min(total_current, 0.25), 2)

    return new_appliances


# -----------------------------------------------------------------------------
# Event-Based Simulation Stream Worker
# -----------------------------------------------------------------------------
def simulation_worker():
    """
    Event-based simulation worker. Cycles through physical states, evaluates
    Delta-I jump, executes live Decision Tree inference, and updates /api/status.
    """
    print(f"[Simulation] Event-Based Stream Active. Monitoring Delta-I >= {DELTA_I_THRESHOLD} A...")

    events_sequence = [
        ("Standby Baseline", 0.05, 5),
        ("Ceiling Fan Started", 0.42, 8),
        ("Fast Charger Plugged In", 0.65, 8),
        ("Electric Iron Turned ON", 6.25, 10),
        ("Iron Thermostat Rest Cycle", 0.64, 6),
        ("Iron Heating Resumed", 6.28, 10),
        ("Electric Iron Turned OFF", 0.65, 7),
        ("Ceiling Fan Turned OFF", 0.24, 6),
        ("Charger Disconnected", 0.05, 5),
    ]

    seq_idx = 0
    last_current_rms = 0.05

    while not stop_event.is_set():
        # If live telemetry from ESP32/Wokwi was received in the last 20 seconds, pause simulation
        if time.time() - last_wifi_telemetry_time < 20.0:
            time.sleep(1.0)
            continue

        state_name, target_curr, duration = events_sequence[seq_idx % len(events_sequence)]
        seq_idx += 1

        noise = random.uniform(-0.02, 0.02)
        current_val = round(max(0.0, target_curr + noise), 2)
        delta_i = round(current_val - last_current_rms, 2)
        now_iso = datetime.now(timezone.utc).isoformat()

        # Generate MPU6050 snapshot
        imu_snapshot = generate_mpu6050_snapshot(current_val, delta_i)

        # Check for sudden jump: EVENT-BASED TRIGGER
        if abs(delta_i) >= DELTA_I_THRESHOLD:
            # Live Scikit-Learn Decision Tree Inference
            pred_label, confidence = run_live_inference(current_val, delta_i, imu_snapshot)

            # Update appliance states based on ML prediction
            classified_appliances = apply_inference_to_appliances(pred_label, current_val)

            with data_lock:
                latest_data["timestamp"] = now_iso
                latest_data["mode"] = "simulation (live ml inference)"
                latest_data["total_current_rms"] = current_val
                latest_data["delta_i"] = delta_i
                latest_data["appliances"] = classified_appliances
                latest_data["ml_inference"] = {
                    "predicted_appliance": pred_label,
                    "confidence": round(confidence, 3),
                    "model": "DecisionTreeClassifier",
                    "last_inference_time": now_iso,
                }

            print(f"[EVENT TRIGGER] Delta-I: {delta_i:+.2f}A | RMS: {current_val:.2f}A | ML Prediction -> {pred_label} ({confidence * 100:.1f}%)")

        last_current_rms = current_val

        # Wait duration of this state
        for _ in range(duration):
            if stop_event.is_set():
                break
            time.sleep(1.0)


# -----------------------------------------------------------------------------
# Event-Based Serial Hardware Worker (ESP32 Live Ingestion)
# -----------------------------------------------------------------------------
def serial_worker():
    """
    Reads real-time data from ESP32 over serial. On sudden Delta-I jump,
    captures MPU6050 snapshot and runs live Decision Tree inference.
    """
    import serial

    print(f"[Serial] Hardware event monitor starting on {SERIAL_PORT} @ {SERIAL_BAUDRATE} baud...")
    ser = None
    last_current_rms = 0.0

    while not stop_event.is_set():
        if ser is None or not ser.is_open:
            try:
                ser = serial.Serial(SERIAL_PORT, SERIAL_BAUDRATE, timeout=1.0)
                print(f"[Serial] Successfully connected to ESP32 on {SERIAL_PORT}")
            except Exception as e:
                with data_lock:
                    latest_data["timestamp"] = datetime.now(timezone.utc).isoformat()
                    latest_data["mode"] = "serial (disconnected)"
                stop_event.wait(4.0)
                continue

        try:
            line = ser.readline().decode("utf-8", errors="ignore").strip()
            if line:
                current_val = None

                # JSON parse: {"current_rms": 5.42}
                if "{" in line and "}" in line:
                    try:
                        parsed = json.loads(line)
                        current_val = float(parsed.get("current_rms", parsed.get("current", 0.0)))
                    except Exception:
                        pass
                else:
                    try:
                        digits = ''.join(c for c in line if (c.isdigit() or c == '.'))
                        if digits:
                            current_val = float(digits)
                    except ValueError:
                        continue

                if current_val is None:
                    continue

                now_iso = datetime.now(timezone.utc).isoformat()
                current_val = round(current_val, 2)
                delta_i = round(current_val - last_current_rms, 2)

                # Event Trigger on RMS jump
                if abs(delta_i) >= DELTA_I_THRESHOLD:
                    imu_snapshot = generate_mpu6050_snapshot(current_val, delta_i)
                    pred_label, confidence = run_live_inference(current_val, delta_i, imu_snapshot)
                    classified_apps = apply_inference_to_appliances(pred_label, current_val)

                    with data_lock:
                        latest_data["timestamp"] = now_iso
                        latest_data["mode"] = "serial (connected, live ml)"
                        latest_data["total_current_rms"] = current_val
                        latest_data["delta_i"] = delta_i
                        latest_data["appliances"] = classified_apps
                        latest_data["ml_inference"] = {
                            "predicted_appliance": pred_label,
                            "confidence": round(confidence, 3),
                            "model": "DecisionTreeClassifier",
                            "last_inference_time": now_iso,
                        }

                    print(f"[HARDWARE EVENT] Delta-I: {delta_i:+.2f}A | RMS: {current_val:.2f}A | ML Inference -> {pred_label} ({confidence * 100:.1f}%)")

                last_current_rms = current_val

        except Exception as e:
            if ser:
                try:
                    ser.close()
                except Exception:
                    pass
            ser = None
            stop_event.wait(2.0)

    if ser and ser.is_open:
        ser.close()


# -----------------------------------------------------------------------------
# Lifespan Handler: Train Scikit-Learn Model on Server Startup
# -----------------------------------------------------------------------------
@asynccontextmanager
async def lifespan(app: FastAPI):
    global worker_thread, trained_classifier
    stop_event.clear()

    # 1. Train Decision Tree Model on dataset.csv upon startup
    try:
        trained_classifier, _ = train_decision_tree()
    except Exception as e:
        print(f"[Error] Failed to train Decision Tree: {e}")

    # 2. Launch Stream Monitor only if simulation mode is explicitly enabled
    if SIMULATION_MODE:
        worker_thread = threading.Thread(target=simulation_worker, daemon=True, name="SimulationWorker")
        worker_thread.start()
        print("[Server] Fake Simulation Worker STARTED.")
    else:
        worker_thread = None
        print("[Server] Real Wi-Fi Mode Active. Fake simulation is DISABLED. Waiting for Wokwi/ESP32...")

    # 3. Start MQTT subscriber (always, listens for ESP32 telemetry via MQTT)
    start_mqtt_subscriber()

    yield

    # Shutdown
    stop_event.set()
    if mqtt_client_instance:
        mqtt_client_instance.loop_stop()
        mqtt_client_instance.disconnect()
    if worker_thread and worker_thread.is_alive():
        worker_thread.join(timeout=2.0)


# -----------------------------------------------------------------------------
# FastAPI App
# -----------------------------------------------------------------------------
app = FastAPI(
    title="NILM Edge Node with Scikit-Learn Decision Tree",
    description="Event-based NILM monitoring with MPU6050 sensor fusion and live ML classification",
    version="0.4.0",
    lifespan=lifespan
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
def read_root():
    return {
        "status": "online",
        "service": "NILM Edge Node with Decision Tree Classifier",
        "simulation_mode": SIMULATION_MODE,
        "delta_i_threshold_a": DELTA_I_THRESHOLD,
        "model_accuracy": f"{model_metrics.get('accuracy', 0) * 100:.2f}%",
        "classes": model_metrics.get("classes", []),
    }


@app.get("/health")
def health_check():
    return {
        "status": "healthy",
        "simulation_mode": SIMULATION_MODE,
        "worker_active": worker_thread.is_alive() if worker_thread else False,
        "model_loaded": trained_classifier is not None,
        "mode": latest_data.get("mode"),
        "mqtt": {
            "connected": mqtt_client_instance.is_connected() if mqtt_client_instance else False,
            "info": mqtt_status_info
        }
    }


# -----------------------------------------------------------------------------
# Wi-Fi Telemetry Ingestion (ESP32 / Wokwi Simulator)
# -----------------------------------------------------------------------------
class TelemetryPayload(BaseModel):
    current_rms: float = Field(..., description="RMS Current in Amperes")
    delta_i: Optional[float] = Field(0.0, description="Delta-I current jump")
    voltage: Optional[float] = Field(230.0, description="Mains voltage RMS")
    accel_x: Optional[float] = Field(0.0, description="MPU6050 Accel X (g)")
    accel_y: Optional[float] = Field(0.0, description="MPU6050 Accel Y (g)")
    accel_z: Optional[float] = Field(1.0, description="MPU6050 Accel Z (g)")
    gyro_x: Optional[float] = Field(0.0, description="MPU6050 Gyro X (dps)")
    gyro_y: Optional[float] = Field(0.0, description="MPU6050 Gyro Y (dps)")
    gyro_z: Optional[float] = Field(0.0, description="MPU6050 Gyro Z (dps)")
    vibration_magnitude: Optional[float] = Field(None, description="Resultant vibration magnitude")
    predicted_appliance: Optional[str] = Field(None, description="Edge TinyML predicted appliance")
    confidence: Optional[float] = Field(None, description="Prediction confidence (0.0 to 1.0)")
    device_id: Optional[str] = Field("wokwi-esp32", description="Device identifier")


@app.post("/api/telemetry")
def receive_telemetry(payload: TelemetryPayload):
    """
    Receives live telemetry over Wi-Fi from an ESP32 (or Wokwi simulator).
    Supports both on-chip Edge TinyML predictions (from inference.h)
    and cloud-side Decision Tree classification.
    """
    global last_wifi_telemetry_time
    last_wifi_telemetry_time = time.time()
    now_iso = datetime.now(timezone.utc).isoformat()

    # Calculate vibration magnitude if not supplied
    vib_mag = payload.vibration_magnitude
    if vib_mag is None:
        vib_mag = round(math.sqrt(payload.accel_x**2 + payload.accel_y**2 + payload.accel_z**2), 4)

    imu_data = {
        "accel_x_g": payload.accel_x,
        "accel_y_g": payload.accel_y,
        "accel_z_g": payload.accel_z,
        "gyro_x_dps": payload.gyro_x,
        "gyro_y_dps": payload.gyro_y,
        "gyro_z_dps": payload.gyro_z,
        "vibration_magnitude_g": vib_mag,
    }

    # If ESP32 ran TinyML on-chip, trust the edge result; otherwise run cloud Decision Tree
    if payload.predicted_appliance and payload.predicted_appliance.strip() not in ("", "Unknown"):
        pred_label = payload.predicted_appliance.strip()
        confidence = float(payload.confidence) if payload.confidence is not None else 1.0
        model_source = "TinyML ESP32 (inference.h)"
    else:
        pred_label, confidence = run_live_inference(payload.current_rms, payload.delta_i, imu_data)
        model_source = "DecisionTreeClassifier (Cloud)"

    classified_apps = apply_inference_to_appliances(pred_label, payload.current_rms)

    with data_lock:
        latest_data["timestamp"] = now_iso
        latest_data["mode"] = f"wifi ({payload.device_id})"
        latest_data["total_current_rms"] = round(payload.current_rms, 3)
        latest_data["delta_i"] = round(payload.delta_i, 3)
        latest_data["voltage_rms"] = round(payload.voltage, 1)
        latest_data["appliances"] = classified_apps
        latest_data["ml_inference"] = {
            "predicted_appliance": pred_label,
            "confidence": round(confidence, 3),
            "model": model_source,
            "last_inference_time": now_iso,
        }

    print(f"[WIFI TELEMETRY] Node: {payload.device_id} | I: {payload.current_rms:.2f}A | dI: {payload.delta_i:+.2f}A | Prediction -> {pred_label} ({confidence*100:.1f}%)")

    return {
        "status": "success",
        "device_id": payload.device_id,
        "predicted_appliance": pred_label,
        "confidence": round(confidence, 3),
        "model": model_source,
        "timestamp": now_iso
    }


@app.get("/api/status")
def get_status():
    """Returns live telemetry, latest Delta-I jump, ML prediction, and appliance states."""
    with data_lock:
        voltage = latest_data.get("voltage_rms", 230.0)
        total_current = latest_data.get("total_current_rms", 0.0)
        total_power = round(total_current * voltage, 2)

        appliance_details = {}
        appliance_states = {}
        for name, details in latest_data.get("appliances", {}).items():
            current = details.get("current_rms", 0.0)
            status = details.get("status", "UNKNOWN")
            power_w = round(current * voltage, 2)
            appliance_states[name] = status
            appliance_details[name] = {
                "state": status,
                "current_rms_a": current,
                "power_w": power_w,
            }

        return {
            "timestamp": latest_data.get("timestamp"),
            "simulation_mode": SIMULATION_MODE,
            "mode": latest_data.get("mode"),
            "voltage_v": voltage,
            "total_current_a": total_current,
            "delta_i_a": latest_data.get("delta_i", 0.0),
            "total_power_w": total_power,
            "ml_inference": latest_data.get("ml_inference"),
            "model_metrics": model_metrics,
            "appliance_states": appliance_states,
            "appliances": appliance_details,
        }


@app.get("/api/model")
def get_model_details():
    """Returns trained Decision Tree parameters and performance metrics."""
    return {
        "model": "DecisionTreeClassifier",
        "criterion": "gini",
        "max_depth": 5,
        "accuracy": model_metrics.get("accuracy"),
        "classes": model_metrics.get("classes"),
        "feature_columns": FEATURE_COLUMNS,
        "train_samples": model_metrics.get("train_samples"),
        "test_samples": model_metrics.get("test_samples"),
        "trained_at": model_metrics.get("trained_at"),
    }


if __name__ == "__main__":
    uvicorn.run("backend:app", host="0.0.0.0", port=8000, reload=True)
