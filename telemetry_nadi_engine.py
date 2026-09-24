"""
Smartwatch Telemetry & Digital Nadi Pariksha (Pulse Diagnostics) Engine
======================================================================
Bridges real-time wearable biometrics (PPG Heart Rate, HRV RMSSD,
Skin Temperature, Sleep Efficiency, and SpO2) with Ayurvedic Tridosha
physiology and classical Nadi Gati (pulse gait) clinical analysis.

Solves GitHub Issue #8: Smart watch connectivity.
"""

from typing import Dict, Any, Optional
from pydantic import BaseModel, Field

class SmartwatchTelemetry(BaseModel):
    device_name: Optional[str] = "Generic Wearable"
    resting_heart_rate: float = Field(..., ge=35.0, le=220.0, description="Resting HR in beats per minute")
    hrv_rmssd: float = Field(..., ge=5.0, le=250.0, description="Heart Rate Variability RMSSD in milliseconds")
    skin_temp_celsius: Optional[float] = Field(36.5, ge=33.0, le=42.0, description="Skin/Body Temperature in Celsius")
    spo2_pct: Optional[float] = Field(98.0, ge=70.0, le=100.0, description="Blood oxygen saturation percentage")
    respiratory_rate: Optional[float] = Field(16.0, ge=6.0, le=40.0, description="Breaths per minute")
    sleep_efficiency_pct: Optional[float] = Field(85.0, ge=10.0, le=100.0, description="Sleep efficiency percentage")
    stress_index: Optional[float] = Field(30.0, ge=0.0, le=100.0, description="Autonomic stress score (0-100)")

def analyze_smartwatch_nadi(telemetry: SmartwatchTelemetry) -> Dict[str, Any]:
    """
    Analyzes raw smartwatch biometric signals and translates them into an
    Ayurvedic Digital Nadi Pariksha report with Tridosha drift and Ojas scores.
    """
    hr = telemetry.resting_heart_rate
    hrv = telemetry.hrv_rmssd
    temp = telemetry.skin_temp_celsius if telemetry.skin_temp_celsius is not None else 36.5
    resp = telemetry.respiratory_rate if telemetry.respiratory_rate is not None else 16.0
    sleep = telemetry.sleep_efficiency_pct if telemetry.sleep_efficiency_pct is not None else 85.0
    stress = telemetry.stress_index if telemetry.stress_index is not None else 30.0

    # 1. Tridosha Physiological Scoring
    # Vata Scoring: Rapid irregularity, elevated stress, low HRV, shallow breathing, fragmented sleep
    vata_points = 20.0
    if hrv < 30.0:
        vata_points += 25.0
    elif hrv < 45.0:
        vata_points += 15.0
    if stress > 55.0:
        vata_points += 25.0
    if resp > 18.0:
        vata_points += 15.0
    if sleep < 75.0:
        vata_points += 15.0

    # Pitta Scoring: Elevated resting tachycardia, elevated body temperature, active metabolism
    pitta_points = 20.0
    if hr > 82.0:
        pitta_points += 30.0
    elif hr > 74.0:
        pitta_points += 15.0
    if temp > 37.0:
        pitta_points += 25.0
    elif temp > 36.6:
        pitta_points += 10.0
    if 40.0 <= stress <= 65.0:
        pitta_points += 15.0

    # Kapha Scoring: Low resting bradycardia, high parasympathetic tone (elevated HRV), slow breathing, deep sleep
    kapha_points = 20.0
    if hr < 62.0:
        kapha_points += 30.0
    elif hr < 70.0:
        kapha_points += 15.0
    if hrv > 65.0:
        kapha_points += 25.0
    elif hrv > 45.0:
        kapha_points += 10.0
    if resp < 14.0:
        kapha_points += 15.0
    if sleep > 88.0 and stress < 30.0:
        kapha_points += 15.0

    # Normalize to 100%
    total_points = vata_points + pitta_points + kapha_points
    vata_pct = round((vata_points / total_points) * 100.0, 1)
    pitta_pct = round((pitta_points / total_points) * 100.0, 1)
    kapha_pct = round((kapha_points / total_points) * 100.0, 1)

    # 2. Determine Classical Nadi Gati (Pulse Movement)
    scores = {"Vata": vata_pct, "Pitta": pitta_pct, "Kapha": kapha_pct}
    sorted_doshas = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    top_dosha, top_val = sorted_doshas[0]
    second_dosha, second_val = sorted_doshas[1]

    if top_val - second_val < 7.0:
        nadi_type = "Sama Nadi (Harmonious / Tri-dosha Balanced Pulse)"
        gati_name = "Sama Gati"
        gati_description = "A smooth, harmonious pulse rhythm exhibiting stability across physical and autonomic axes."
    elif top_dosha == "Vata":
        nadi_type = "Vata Nadi (Sarpa Gati / Snake-like Pulse)"
        gati_name = "Sarpa Gati (Serpentine)"
        gati_description = "A rapid, thin, variable pulse indicating elevated autonomic nervous arousal, heightened mental stress, or muscular tension."
    elif top_dosha == "Pitta":
        nadi_type = "Pitta Nadi (Manduka Gati / Frog-like Bounding Pulse)"
        gati_name = "Manduka Gati (Froggish)"
        gati_description = "A forceful, sharp, bounding pulse reflecting elevated metabolic thermogenesis, cardiovascular vigor, or systemic inflammatory heat."
    else:
        nadi_type = "Kapha Nadi (Hamsa Gati / Swan-like Graceful Pulse)"
        gati_name = "Hamsa Gati (Swan-like)"
        gati_description = "A deep, calm, steady, regular pulse signifying parasympathetic dominance, slower metabolic rhythm, and anabolic stability."

    # 3. Autonomic Tone (Sympathovagal Balance)
    if stress >= 60.0 or (hrv < 25.0 and hr > 80.0):
        autonomic_state = "Sympathetic Dominance (Fight-or-Flight Overdrive)"
        autonomic_interpretation = "Elevated cortisol and sympathetic tone. The body is in an acute expenditure state; grounding interventions needed."
    elif stress <= 25.0 and hrv >= 60.0:
        autonomic_state = "Parasympathetic Dominance (Rest & Restoration)"
        autonomic_interpretation = "Excellent vagal tone and tissue recovery. Anabolic and cellular repair mechanisms are actively operating."
    else:
        autonomic_state = "Homeostatic Balance (Adaptive Equilibrium)"
        autonomic_interpretation = "Healthy balance between work energy expenditure and autonomic recuperation."

    # 4. Ojas Vitality Index (Immune & Energy Resilience: 0 to 100)
    # Formulated from normalized HRV resilience, sleep restorative efficiency, and stress buffering
    hrv_component = min(100.0, (hrv / 75.0) * 100.0)
    sleep_component = min(100.0, sleep)
    stress_buffer = max(0.0, 100.0 - stress)
    ojas_score = round((0.4 * hrv_component) + (0.4 * sleep_component) + (0.2 * stress_buffer), 1)

    if ojas_score >= 80.0:
        ojas_status = "Optimal Ojas (Vibrant Immunity & Vitality)"
    elif ojas_score >= 60.0:
        ojas_status = "Moderate Ojas (Adequate Vital Reserve)"
    else:
        ojas_status = "Depleted Ojas (Prone to Fatigue & Low Resilience)"

    # 5. Tailored Biometric Action Plan
    recommendations = []
    if top_dosha == "Vata" or vata_pct > 38.0:
        recommendations.append("Practice Nadi Shodhana (Alternate Nostril Pranayama) for 10 minutes to elevate vagal HRV.")
        recommendations.append("Ensure sleep prior to 10:30 PM to counteract sympathetic elevation and nervous exhaustion.")
        recommendations.append("Warm Abhyanga (sesame oil massage) recommended to ground neuromuscular agitation.")
    if top_dosha == "Pitta" or pitta_pct > 38.0:
        recommendations.append("Incorporate cooling Shitali/Sitkari breathwork to moderate resting heart rate and thermic load.")
        recommendations.append("Hydrate with tender coconut water or infused coriander-fennel water to soothe Pitta heat.")
        recommendations.append("Avoid intense midday cardio until resting heart rate normalizes below 75 BPM.")
    if top_dosha == "Kapha" or kapha_pct > 38.0:
        recommendations.append("Engage in brisk morning aerobic activity (Surya Namaskar) to stimulate sluggish circulation.")
        recommendations.append("Incorporate warming digestive teas (ginger, cinnamon, black pepper) to kindle Agni.")

    return {
        "status": "success",
        "telemetry_source": telemetry.device_name,
        "biometrics_summary": {
            "resting_heart_rate_bpm": hr,
            "hrv_rmssd_ms": hrv,
            "skin_temp_celsius": temp,
            "spo2_pct": telemetry.spo2_pct,
            "respiratory_rate": resp,
            "sleep_efficiency_pct": sleep,
            "stress_index": stress
        },
        "digital_nadi": {
            "nadi_type": nadi_type,
            "gati_classification": gati_name,
            "gati_description": gati_description,
            "dominant_dosha": top_dosha
        },
        "dosha_drift_matrix": {
            "vata_percentage": vata_pct,
            "pitta_percentage": pitta_pct,
            "kapha_percentage": kapha_pct
        },
        "autonomic_nervous_system": {
            "state": autonomic_state,
            "interpretation": autonomic_interpretation
        },
        "ojas_vitality_index": {
            "score": ojas_score,
            "status": ojas_status
        },
        "biometric_clinical_guidance": recommendations
    }

def get_sample_telemetry(profile_type: str = "vata_stress") -> Dict[str, Any]:
    """
    Returns realistic simulated smartwatch telemetry profiles for testing and clinical demo.
    """
    profiles = {
        "vata_stress": {
            "device_name": "Apple Watch Series 9",
            "resting_heart_rate": 78.0,
            "hrv_rmssd": 22.0,
            "skin_temp_celsius": 36.2,
            "spo2_pct": 98.0,
            "respiratory_rate": 19.0,
            "sleep_efficiency_pct": 68.0,
            "stress_index": 72.0
        },
        "pitta_heat": {
            "device_name": "Garmin Fenix 7",
            "resting_heart_rate": 88.0,
            "hrv_rmssd": 38.0,
            "skin_temp_celsius": 37.4,
            "spo2_pct": 99.0,
            "respiratory_rate": 17.0,
            "sleep_efficiency_pct": 82.0,
            "stress_index": 52.0
        },
        "kapha_calm": {
            "device_name": "Fitbit Charge 6",
            "resting_heart_rate": 56.0,
            "hrv_rmssd": 78.0,
            "skin_temp_celsius": 36.4,
            "spo2_pct": 99.0,
            "respiratory_rate": 12.0,
            "sleep_efficiency_pct": 94.0,
            "stress_index": 18.0
        }
    }
    return profiles.get(profile_type, profiles["vata_stress"])
