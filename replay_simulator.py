"""
Standalone Hardware Replay Simulator for AIoT NILM
Emulates full ESP32 telemetry with multi-sensor fusion (CT clamp + MPU6050)
over MQTT to test backend and Next.js frontend without physical hardware or Wokwi.
"""

import time
import json
import math
import random
import argparse
import paho.mqtt.client as mqtt

BROKER = "broker.hivemq.com"
PORT = 1883
TOPIC_TELEMETRY = "nilm/pushkal/telemetry"
TOPIC_STATUS = "nilm/pushkal/status"
TOPIC_CONTROL = "nilm/pushkal/control"
TOPIC_ACK = "nilm/pushkal/ack"
DEVICE_ID = "esp32-nilm-sim"

breaker_tripped = False


def on_connect(client, userdata, flags, rc, properties=None):
    print(f"[Simulator] Connected to {BROKER}:{PORT} (rc={rc})")
    # Publish Birth message
    birth_payload = {
        "device_id": DEVICE_ID,
        "status": "ONLINE",
        "firmware": "2.1.0",
        "mode": "PYTHON_REPLAY_SIMULATOR",
        "free_heap": 182400
    }
    client.publish(TOPIC_STATUS, json.dumps(birth_payload), retain=True)
    client.subscribe(TOPIC_CONTROL)
    print(f"[Simulator] Subscribed to control topic: {TOPIC_CONTROL}")


def on_message(client, userdata, msg):
    global breaker_tripped
    try:
        payload_str = msg.payload.decode("utf-8")
        print(f"\n[Simulator] Received control command on {msg.topic}: {payload_str}")
        data = json.loads(payload_str) if "{" in payload_str else {"action": payload_str}

        action = data.get("action", "").upper()
        cmd_id = data.get("command_id", f"sim_{int(time.time())}")

        if "TRIP" in action or "SHED" in action:
            breaker_tripped = True
            print(">>> [SIMULATOR BREAKER] Relay TRIPPED: Current set to 0.0A! <<<")
        elif "RESET" in action or "ARM" in action:
            breaker_tripped = False
            print(">>> [SIMULATOR BREAKER] Relay ARMED: Normal load restored. <<<")

        ack_payload = {
            "command_id": cmd_id,
            "action": action,
            "status": "SUCCESS",
            "breaker_status": "TRIPPED" if breaker_tripped else "ARMED",
            "execution_time_ms": 3,
            "timestamp": int(time.time())
        }
        client.publish(TOPIC_ACK, json.dumps(ack_payload))
        print(f"[Simulator] Published execution ACK: {ack_payload}")
    except Exception as e:
        print(f"[Simulator] Error processing control message: {e}")


def build_telemetry(seq: int, current_rms: float, delta_i: float, app_name: str, state_event: str,
                    vib_mag: float, gy: float, latency_us: int) -> dict:
    if breaker_tripped:
        current_rms = 0.0
        delta_i = -0.5
        app_name = "Standby"
        state_event = "BREAKER_TRIPPED"

    ax = round(random.gauss(0.01, 0.005), 4)
    ay = round(random.gauss(0.01, 0.005), 4)
    az = round(math.sqrt(max(0.0, vib_mag**2 - ax**2 - ay**2)), 4)
    gx = round(random.uniform(-0.1, 0.1), 2)
    gz = round(random.uniform(-0.1, 0.1), 2)

    return {
        "device_id": DEVICE_ID,
        "seq": seq,
        "uptime_seconds": seq * 3,
        "current_rms": round(current_rms, 2),
        "delta_i": round(delta_i, 2),
        "voltage": 230.0,
        "accel_x": ax,
        "accel_y": ay,
        "accel_z": az,
        "gyro_x": gx,
        "gyro_y": round(gy, 2),
        "gyro_z": gz,
        "vibration_magnitude": round(vib_mag, 4),
        "predicted_appliance": app_name,
        "state_event": state_event,
        "confidence": 1.0,
        "inference_latency_us": latency_us,
        "breaker_status": "TRIPPED" if breaker_tripped else "ARMED",
        "free_heap_bytes": 182000 - (seq % 100) * 10
    }


def run_simulation(loop_forever: bool = False):
    client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2)
    client.on_connect = on_connect
    client.on_message = on_message

    lwt_payload = {"device_id": DEVICE_ID, "status": "OFFLINE", "reason": "SIMULATOR_STOPPED"}
    client.will_set(TOPIC_STATUS, json.dumps(lwt_payload), qos=1, retain=True)

    client.connect(BROKER, PORT, 60)
    client.loop_start()
    time.sleep(1.0)

    # Workflow script: realistic sequence testing positive ON & negative OFF resolution
    steps = [
        # (name, current, delta_i, appliance, state_event, vib_mag, gy, latency_us, duration_sec)
        ("Standby Baseline", 0.05, 0.00, "Standby", "STANDBY", 1.002, 0.02, 12, 4),
        ("Ceiling Fan Turn ON", 0.42, 0.37, "Ceiling Fan", "ON_TRANSITION", 1.025, -1.85, 14, 6),
        ("Fast Charger Plugged In", 0.66, 0.24, "Fast Charger", "ON_TRANSITION", 1.028, -1.70, 13, 6),
        ("Electric Iron Turn ON", 6.22, 5.56, "Electric Iron", "ON_TRANSITION", 1.030, -0.05, 15, 8),
        ("High Vibration Alert on Fan", 6.25, 0.03, "Electric Iron", "STEADY_STATE", 1.245, -2.10, 14, 6),
        ("Electric Iron Turn OFF (Negative Step)", 0.67, -5.55, "Electric Iron", "OFF_TRANSITION", 1.028, -1.75, 14, 6),
        ("Ceiling Fan Turn OFF (Negative Step)", 0.25, -0.42, "Ceiling Fan", "OFF_TRANSITION", 1.004, 0.04, 13, 6),
        ("Fast Charger Disconnected", 0.05, -0.20, "Fast Charger", "OFF_TRANSITION", 1.001, 0.02, 12, 4),
    ]

    seq = 1
    print("\n=======================================================")
    print(" AIoT NILM Python Replay Simulator Active")
    print(f" Publishing to {BROKER}:{PORT} -> {TOPIC_TELEMETRY}")
    print("=======================================================\n")

    try:
        while True:
            for step_name, curr, delta, app, event, vib, gy, lat, dur in steps:
                pkt = build_telemetry(seq, curr, delta, app, event, vib, gy, lat)
                payload_json = json.dumps(pkt)
                client.publish(TOPIC_TELEMETRY, payload_json)
                print(f"[PUBLISH #{seq:03d}] {step_name:<34} | I: {pkt['current_rms']:.2f}A (dI: {pkt['delta_i']:+.2f}A) | {pkt['predicted_appliance']} ({event})")
                seq += 1
                time.sleep(dur)

            if not loop_forever:
                print("\n[Simulator] Sequence completed successfully.")
                break
    except KeyboardInterrupt:
        print("\n[Simulator] Interrupted by user.")
    finally:
        client.loop_stop()
        client.disconnect()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="AIoT NILM Hardware Replay Simulator")
    parser.add_argument("--continuous", action="store_true", help="Loop appliance sequence continuously")
    args = parser.parse_args()
    run_simulation(loop_forever=args.continuous)
