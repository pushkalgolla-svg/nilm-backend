# AIoT NILM Multi-Sensor Dataset & Empirical Verification Documentation

## 1. Dataset Overview & Class Demarcation

This dataset captures synchronized multi-modal features combining electrical current transients and high-frequency kinetic vibration signals from an MPU-6050 6-DOF IMU.

- **Classes:** 4 Target States
  1. `Ceiling Fan` (Class ID 0): ~60W BLDC/induction motor, active RMS current $0.35\text{--}0.42\text{ A}$, characterized by negative angular velocity rotational harmonic ($\omega_y \le -0.58^\circ/\text{s}$).
  2. `Electric Iron` (Class ID 1): High-power resistive heating element ($5.40\text{--}5.85\text{ A}$, ~1250W), zero kinetic rotation ($|\omega| \approx 0$).
  3. `Fast Charger` (Class ID 2): GaN SMPS adapter ($0.20\text{--}0.26\text{ A}$, ~65W), minimal vibration signature ($|a| \approx 1.00g$).
  4. `Standby` (Class ID 3): Quiescent household background noise ($< 0.10\text{ A}$, static IMU).

- **Total Samples:** 100 balanced laboratory samples (25 per class).
- **Split:** 75% Training (75 samples), 25% Stratified Holdout Test (25 samples).

---

## 2. Decision Tree Architecture & Edge Footprint

The decision tree was trained using Scikit-Learn `DecisionTreeClassifier(max_depth=5, criterion='gini')` and transpiled directly to C99 (`inference.h`) via `export_tinyml.py`:

```c
if (features[1] <= 0.112500f) {
    return 3; // Standby
} else {
    if (features[1] <= 2.752500f) {
        if (features[6] <= -0.585000f) {
            return 0; // Ceiling Fan
        } else {
            return 2; // Fast Charger
        }
    } else {
        return 1; // Electric Iron
    }
}
```

### Empirical Edge Benchmark Metrics (ESP32 Xtensa Dual-Core 240MHz):
- **Inference Latency:** $11\text{--}16\ \mu\text{s}$ (measured via hardware timer `micros()`).
- **RAM Footprint:** $< 1\text{ KB}$ (Zero dynamic `malloc()` allocations; static stack-allocated array).
- **Flash Storage:** $< 5\text{ KB}$ (C99 compiled binary footprint).
- **Decision Boundaries:**
  - $\Delta I \le 0.1125\text{ A} \rightarrow$ Standby.
  - $\Delta I > 2.7525\text{ A} \rightarrow$ Electric Iron.
  - $0.1125\text{ A} < \Delta I \le 2.7525\text{ A}$ & $\omega_y \le -0.585^\circ/\text{s} \rightarrow$ Ceiling Fan.
  - $0.1125\text{ A} < \Delta I \le 2.7525\text{ A}$ & $\omega_y > -0.585^\circ/\text{s} \rightarrow$ Fast Charger.

---

## 3. Honest Engineering Limitations & Real-World Generalization

> [!IMPORTANT]
> **Academic Integrity Statement on Reported 100% Accuracy:**
> The reported 100.00% classification accuracy was achieved on a 25-sample stratified holdout set under controlled laboratory conditions where class electrical boundaries are distinctly separated. This metric demonstrates optimal model convergence and zero underfitting, but must **not** be presented as evidence of perfect field performance across non-laboratory environments.

### Real-World Environmental Variables Not Captured in the 100-Sample Baseline:
1. **Mains Voltage Fluctuations:** Indian electrical grid supplies frequently fluctuate between $190\text{V}$ and $250\text{V}$, causing proportional shifts in resistive load current ($I = V/R$).
2. **Thermal Resistance Creep:** Heating appliances (e.g. Electric Iron nichrome wire) exhibit positive temperature coefficients, where cold start draw is significantly higher than steady-state hot draw.
3. **Multi-Speed Fan Settings:** Ceiling fans running at Speed 1 draw lower current ($~0.22\text{ A}$) and emit lower vibration than at Speed 5 ($~0.42\text{ A}$), which could overlap with fast chargers without gyro calibration.
4. **Sensor Mounting Mechanics:** MPU-6050 accelerometer and gyroscope readings vary depending on whether the sensor is rigidly bolted to the motor casing or taped to a dampening plastic shroud.

### Recommended Transfer-Learning Pipeline for Production:
1. Collect continuous 24-hour real-world data logs across target electrical environments using the firmware's SQLite telemetry logger.
2. Augment the training dataset with intermediate voltage steps ($210\text{V}, 220\text{V}, 230\text{V}, 240\text{V}$) and varying fan speed settings.
3. Retrain the Decision Tree with expanded datasets ($N > 2000$) and re-run `export_tinyml.py` to regenerate `inference.h`.
