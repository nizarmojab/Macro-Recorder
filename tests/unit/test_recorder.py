"""
Tests unitaires pour :
- OmniParser.find_element_at  (match mathématique bbox)
- Format JSON sauvegardé par SmartRecorder
"""
import json
import pytest
from unittest.mock import patch


# ── Helpers ───────────────────────────────────────────────────────────────────

def _make_element(label, elem_type, bbox, confidence=0.9):
    from vision.grounding.omniparser import UIElement
    x1, y1, x2, y2 = bbox
    return UIElement(
        label=label,
        element_type=elem_type,
        x=(x1 + x2) // 2,
        y=(y1 + y2) // 2,
        width=x2 - x1,
        height=y2 - y1,
        confidence=confidence,
        bbox=bbox,
    )


def _make_parser():
    """Crée un OmniParser sans déclencher le chargement du modèle."""
    from vision.grounding.omniparser import OmniParser
    parser = OmniParser.__new__(OmniParser)
    parser._model = None
    parser._available = False
    return parser


# ── find_element_at ───────────────────────────────────────────────────────────

class TestFindElementAt:

    def test_returns_element_when_coords_inside_bbox(self):
        parser = _make_parser()
        elements = [
            _make_element("Submit", "button", (100, 200, 300, 240)),
            _make_element("Cancel", "button", (350, 200, 550, 240)),
        ]
        with patch.object(parser, "parse", return_value=elements):
            result = parser.find_element_at("fake.png", 200, 220)

        assert result is not None
        assert result.label == "Submit"

    def test_returns_none_when_no_bbox_matches(self):
        parser = _make_parser()
        elements = [
            _make_element("Submit", "button", (100, 200, 300, 240)),
        ]
        with patch.object(parser, "parse", return_value=elements):
            result = parser.find_element_at("fake.png", 500, 500)

        assert result is None

    def test_returns_none_when_no_elements(self):
        parser = _make_parser()
        with patch.object(parser, "parse", return_value=[]):
            result = parser.find_element_at("fake.png", 200, 220)

        assert result is None

    def test_boundary_coords_match(self):
        """Un clic exactement sur le bord de la bbox doit matcher."""
        parser = _make_parser()
        elements = [
            _make_element("Edge", "button", (100, 100, 200, 200)),
        ]
        with patch.object(parser, "parse", return_value=elements):
            result = parser.find_element_at("fake.png", 100, 100)

        assert result is not None
        assert result.label == "Edge"

    def test_returns_first_matching_element(self):
        """Si plusieurs bbox se chevauchent, retourne le premier match."""
        parser = _make_parser()
        elements = [
            _make_element("First",  "button", (100, 100, 300, 300)),
            _make_element("Second", "input",  (150, 150, 250, 250)),
        ]
        with patch.object(parser, "parse", return_value=elements):
            result = parser.find_element_at("fake.png", 200, 200)

        assert result is not None
        assert result.label == "First"


# ── Format JSON ───────────────────────────────────────────────────────────────

class TestRecordingJsonFormat:

    def test_click_event_has_required_fields(self, tmp_path):
        """Chaque événement click doit avoir label, elem_type, confidence, annotated."""
        recording_data = {
            "workflow_name": "test",
            "task_description": "test",
            "started_at": "2026-01-01T00:00:00",
            "annotated": True,
            "events": [
                {
                    "event_type": "click",
                    "raw_x": 200,
                    "raw_y": 220,
                    "label": "Submit",
                    "elem_type": "button",
                    "confidence": 0.95,
                    "annotated": True,
                    "screenshot_path": "data/captures/test.png",
                    "timestamp": 1234567890.0,
                    "key": "Button.left",
                    "text": None,
                    "semantic_label": "Button: Submit",
                    "intent": "Click on Submit",
                }
            ],
        }
        path = tmp_path / "recording.json"
        path.write_text(json.dumps(recording_data))

        with open(path) as f:
            data = json.load(f)

        clicks = [e for e in data["events"] if e["event_type"] == "click"]
        assert len(clicks) > 0
        for event in clicks:
            assert "label" in event
            assert "elem_type" in event
            assert "confidence" in event
            assert event["label"] != ""
            assert event["annotated"] is True

    def test_annotated_flag_is_true_for_annotated_recording(self, tmp_path):
        recording_data = {
            "workflow_name": "test",
            "task_description": "test",
            "started_at": "2026-01-01T00:00:00",
            "annotated": True,
            "events": [],
        }
        path = tmp_path / "recording.json"
        path.write_text(json.dumps(recording_data))

        with open(path) as f:
            data = json.load(f)

        assert data["annotated"] is True

    def test_event_has_intent_for_replay_engine(self, tmp_path):
        """Le replay engine lit event['intent'] — il ne doit pas être vide."""
        recording_data = {
            "workflow_name": "test",
            "task_description": "",
            "started_at": "2026-01-01T00:00:00",
            "annotated": True,
            "events": [
                {
                    "event_type": "click",
                    "raw_x": 200, "raw_y": 220,
                    "label": "Login",
                    "elem_type": "button",
                    "confidence": 0.9,
                    "annotated": True,
                    "screenshot_path": "data/captures/test.png",
                    "timestamp": 1234567890.0,
                    "key": "Button.left",
                    "text": None,
                    "semantic_label": "Button: Login",
                    "intent": "Click on Login",
                }
            ],
        }
        path = tmp_path / "recording.json"
        path.write_text(json.dumps(recording_data))

        with open(path) as f:
            data = json.load(f)

        annotated = [
            e for e in data["events"]
            if e.get("intent") and e["intent"] != "unknown"
        ]
        assert len(annotated) > 0
