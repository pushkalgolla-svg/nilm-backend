# NILM Backend

Non-Intrusive Load Monitoring (NILM) API service built with FastAPI, PySerial, and Scikit-Learn.

## Features
- **FastAPI Core**: High performance, asynchronous API with automatic OpenAPI documentation.
- **Simulation Mode**: Built-in background thread generating realistic RMS current draw and appliance duty cycles for 3 appliances (Refrigerator, Air Conditioner, Microwave).
- **Serial Mode Placeholder**: Background worker prepared to ingest real-time hardware telemetry via pyserial with auto-reconnection.
- **Interactive Swagger Docs**: Available at http://localhost:8000/docs.

## API Endpoints
- `GET /api/status`: Returns live total power draw, active voltage, total current, and per-appliance state & power.
- `GET /api/current`: Returns raw RMS current telemetry.
- `GET /health`: System and worker thread health check.
- `GET /`: Service metadata.

## Running in Antigravity IDE
- **Debug / Run**: Press `F5` or use the Run & Debug view to launch **FastAPI: Run NILM Backend (Simulation)**.
- **Terminal**: Run `.\.venv\Scripts\python.exe backend.py`
