"""
Unit Test Suite for AIoT NILM Core Analytics & Physics Equations
Verifies:
- I_RMS calculation
- Active Power P = 230 * I_RMS
- Riemann Energy Integration E(kWh) = sum(P_k * dt_k) / 3,600,000
- Utility Time-of-Day (ToD) tariff boundaries (Normal, Peak, Off-Peak)
- Carbon Emission factor (0.82 kg/kWh)
- Continuous Health Score H = max(55, 100 - max(0, |a| - 1.10) * 220)
- Appliance duty cycle calculation
"""

import math
import unittest
from datetime import datetime


def calculate_irms(samples: list) -> float:
    """I_RMS = sqrt((1/N) * sum(i[n]^2))"""
    if not samples:
        return 0.0
    sum_sq = sum(s ** 2 for s in samples)
    return math.sqrt(sum_sq / len(samples))


def calculate_active_power(irms: float, voltage: float = 230.0) -> float:
    """P = V * I_RMS"""
    return round(voltage * irms, 2)


def calculate_energy_kwh(power_w: float, dt_sec: float) -> float:
    """E(kWh) = (P * dt) / 3,600,000"""
    return (power_w * dt_sec) / 3600000.0


def calculate_carbon_kg(energy_kwh: float) -> float:
    """Carbon kg = energy_kwh * 0.82 (CEA India factor)"""
    return round(energy_kwh * 0.82, 4)


def get_tod_tariff_rate(hour: int) -> tuple:
    """
    Tariff windows:
    Peak: 18:00 to 22:00 -> Rs 10.00
    Normal: 06:00 to 18:00 -> Rs 7.50
    Off-Peak: 22:00 to 06:00 -> Rs 5.00
    """
    if 18 <= hour < 22:
        return "PEAK", 10.00
    elif 22 <= hour or hour < 6:
        return "OFF-PEAK", 5.00
    else:
        return "NORMAL", 7.50


def calculate_health_score(vib_mag: float) -> int:
    """H = max(55, 100 - max(0, |a| - 1.10) * 220)"""
    excess = max(0.0, vib_mag - 1.10)
    return int(max(55, round(100.0 - excess * 220.0)))


def calculate_duty_cycle(active_sec: float, session_sec: float) -> float:
    """Duty cycle % = (active_sec / session_sec) * 100"""
    if session_sec <= 0:
        return 0.0
    return round(min(100.0, (active_sec / session_sec) * 100.0), 1)


class TestNILMCorePhysicsAndAnalytics(unittest.TestCase):

    def test_irms_pure_sine_wave(self):
        """A pure sine wave of peak 10.0A must have RMS = 10 / sqrt(2) ~= 7.071A."""
        N = 1000
        peak = 10.0
        samples = [peak * math.sin(2 * math.pi * i / N) for i in range(N)]
        rms = calculate_irms(samples)
        self.assertAlmostEqual(rms, peak / math.sqrt(2), places=2)

    def test_active_power(self):
        """P = 230V * 5.6A (Iron) = 1288W."""
        irms = 5.6
        p = calculate_active_power(irms, 230.0)
        self.assertEqual(p, 1288.0)

    def test_energy_riemann_sum(self):
        """1000W running for 1 hour (3600s) must equal exactly 1.0 kWh."""
        e = calculate_energy_kwh(1000.0, 3600.0)
        self.assertAlmostEqual(e, 1.0, places=5)

    def test_carbon_emission_factor(self):
        """10 kWh * 0.82 kg/kWh = 8.2 kg CO2."""
        co2 = calculate_carbon_kg(10.0)
        self.assertEqual(co2, 8.2)

    def test_tod_tariff_boundaries(self):
        """Test exact hour thresholds for Peak, Normal, and Off-Peak."""
        # 05:59 -> Off-Peak (Rs 5.00)
        slot, rate = get_tod_tariff_rate(5)
        self.assertEqual(slot, "OFF-PEAK")
        self.assertEqual(rate, 5.00)

        # 06:00 -> Normal (Rs 7.50)
        slot, rate = get_tod_tariff_rate(6)
        self.assertEqual(slot, "NORMAL")
        self.assertEqual(rate, 7.50)

        # 17:59 -> Normal (Rs 7.50)
        slot, rate = get_tod_tariff_rate(17)
        self.assertEqual(slot, "NORMAL")
        self.assertEqual(rate, 7.50)

        # 18:00 -> Peak (Rs 10.00)
        slot, rate = get_tod_tariff_rate(18)
        self.assertEqual(slot, "PEAK")
        self.assertEqual(rate, 10.00)

        # 21:59 -> Peak (Rs 10.00)
        slot, rate = get_tod_tariff_rate(21)
        self.assertEqual(slot, "PEAK")
        self.assertEqual(rate, 10.00)

        # 22:00 -> Off-Peak (Rs 5.00)
        slot, rate = get_tod_tariff_rate(22)
        self.assertEqual(slot, "OFF-PEAK")
        self.assertEqual(rate, 5.00)

    def test_health_score_continuity(self):
        """Test unified Health Score formula across vibration regimes."""
        # 1.00g (Normal gravity) -> 100
        self.assertEqual(calculate_health_score(1.00), 100)
        # 1.10g (Boundary of normal band) -> 100
        self.assertEqual(calculate_health_score(1.10), 100)
        # 1.20g (Bearing wear onset) -> 100 - (0.10 * 220) = 78
        self.assertEqual(calculate_health_score(1.20), 78)
        # 1.30g (Severe vibration) -> 100 - (0.20 * 220) = 56
        self.assertEqual(calculate_health_score(1.30), 56)
        # 1.50g (Extreme anomaly clamped at floor 55) -> 55
        self.assertEqual(calculate_health_score(1.50), 55)

    def test_duty_cycle(self):
        """30 minutes active in a 60 minute session = 50.0%."""
        dc = calculate_duty_cycle(1800.0, 3600.0)
        self.assertEqual(dc, 50.0)


if __name__ == "__main__":
    unittest.main()
