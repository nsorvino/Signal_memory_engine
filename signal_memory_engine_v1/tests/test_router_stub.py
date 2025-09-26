import json

import pytest

from agents.router_stub import log_routing_decision, route_agent


def test_axis_high_emotion():
    decision = route_agent(
        "Why is my team so tense?",
        emotional_tone=0.9,
        signal_type="relational",
        drift_score=0.0,
    )
    assert decision["selected_agent"] == "Axis"
    assert "High emotional tone" in decision["reason"]


def test_m_compliance():
    decision = route_agent(
        "Is this legally allowed?",
        emotional_tone=0.3,
        signal_type="compliance",
        drift_score=0.0,
    )
    assert decision["selected_agent"] == "M"
    assert "Compliance signal" in decision["reason"]


def test_m_high_drift():
    # Drift score rule routes to M if > 0.5
    decision = route_agent(
        "Data drift detected",
        emotional_tone=0.1,
        signal_type="relational",
        drift_score=0.6,
    )
    assert decision["selected_agent"] == "M"
    assert "High drift score" in decision["reason"]


@pytest.mark.parametrize("signal", ["relational", "biometric"])
def test_oria_signal_types(signal):
    decision = route_agent("Sample query", emotional_tone=0.1, signal_type=signal, drift_score=0.0)
    assert decision["selected_agent"] == "Oria"
    assert f"Signal type '{signal}'" in decision["reason"]


def test_selah_fallback():
    decision = route_agent(
        "Unknown signal", emotional_tone=0.1, signal_type="unknown", drift_score=0.0
    )
    assert decision["selected_agent"] == "Selah"
    assert "Fallback routing" in decision["reason"]


def test_invalid_tone_type():
    # Non-numeric emotional tone should fallback
    decision = route_agent(
        "Invalid tone type",
        emotional_tone="bad",
        signal_type="relational",
        drift_score=0.0,
    )
    assert decision["selected_agent"] == "Selah"
    assert "Invalid score input" in decision["reason"]


def test_tone_out_of_range():
    # Emotional tone <0 or >1 should fallback
    decision = route_agent(
        "Tone out of range",
        emotional_tone=-0.1,
        signal_type="relational",
        drift_score=0.0,
    )
    assert decision["selected_agent"] == "Selah"
    assert "Emotional tone out of range" in decision["reason"]


def test_drift_out_of_range():
    # Drift score <0 or >1 should fallback
    decision = route_agent(
        "Drift out of range",
        emotional_tone=0.1,
        signal_type="relational",
        drift_score=1.5,
    )
    assert decision["selected_agent"] == "Selah"
    assert "Drift score out of range" in decision["reason"]


def test_invalid_signal_type():
    # Non-string signal_type should fallback
    decision = route_agent(
        "Invalid signal type", emotional_tone=0.1, signal_type=None, drift_score=0.0
    )
    assert decision["selected_agent"] == "Selah"
    assert "Invalid signal type" in decision["reason"]


def test_logging(tmp_path):
    # Ensure logging appends a timestamped entry
    decision = {"selected_agent": "Axis", "reason": "High emotional tone"}
    log_file = tmp_path / "router_log.jsonl"
    log_routing_decision(decision, logfile=str(log_file))

    lines = log_file.read_text().splitlines()
    assert len(lines) == 1, "Expected one log entry"
    entry = json.loads(lines[-1])
    assert entry["selected_agent"] == "Axis"
    assert entry["reason"] == "High emotional tone"
    assert "timestamp" in entry, "Log entry must include a timestamp"


if __name__ == "__main__":
    pytest.main([__file__])
