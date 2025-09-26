# sensors/biometric.py
"""
Simulated biometric sampling, including future metrics (GSR) and auto-tagged emotional state.
"""
import os
import random
from typing import Any

import openai

# Load OpenAI key for emotion classification
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")


def sample_all_signals() -> dict[str, Any]:
    """
    Return a dictionary of current biometric readings, including:
      - HRV (ms)
      - Skin temperature (°C)
      - Blink rate (bpm)
      - GSR (µS)
      - emotion_label (auto-tagged)
    """
    # Simulate sensor readings
    hrv = random.uniform(30.0, 100.0)
    temperature = random.uniform(35.5, 37.5)
    blink_rate = random.uniform(10.0, 20.0)
    # Future biometric: galvanic skin response (µS)
    gsr = random.uniform(0.1, 10.0)

    # Auto-tag emotion based on readings
    emotion_label = _detect_emotion_label(hrv, temperature, blink_rate, gsr)

    return {
        "hrv": hrv,
        "temperature": temperature,
        "blink_rate": blink_rate,
        "gsr": gsr,
        "emotion_label": emotion_label,
    }


def _detect_emotion_label(hrv: float, temperature: float, blink_rate: float, gsr: float) -> str:
    """
    Use a lightweight OpenAI prompt to classify the user's emotional state given biometric signals.
    Returns a single-label string (e.g., 'calm', 'anxious', 'stressed', 'happy').
    """
    if not OPENAI_API_KEY:
        return "unknown"

    openai.api_key = OPENAI_API_KEY
    prompt = (
        "Based on the following physiological readings, choose one of: calm, anxious, stressed, happy."
        f"\nHRV: {hrv:.1f} ms"
        f"\nSkin Temp: {temperature:.1f} °C"
        f"\nBlink Rate: {blink_rate:.1f} bpm"
        f"\nGSR: {gsr:.2f} µS"
    )
    try:
        resp = openai.Completion.create(
            engine="text-davinci-003",
            prompt=prompt,
            max_tokens=5,
            temperature=0.0,
        )
        label = resp.choices[0].text.strip().lower()
        # sanitize to allowed labels
        for allowed in ("calm", "anxious", "stressed", "happy"):
            if allowed in label:
                return allowed
    except Exception:
        pass
    return "unknown"
