"""
Unit tests for the vision capture and VLM client modules.
"""
import json
import os
import pytest
from pathlib import Path
from unittest.mock import patch, MagicMock

os.environ.setdefault("VLM_BASE_URL", "http://localhost:8000/v1")
os.environ.setdefault("VLM_MODEL", "Qwen/Qwen2.5-VL-7B-Instruct")


class TestScreenCapture:
    def test_capture_returns_path(self, tmp_path):
        with patch("mss.mss") as mock_mss:
            mock_sct = MagicMock()
            mock_sct.monitors = [None, {"top": 0, "left": 0, "width": 1920, "height": 1080}]
            mock_sct.grab.return_value = MagicMock(rgb=b"\x00" * 100, size=(10, 10))
            mock_mss.return_value.__enter__ = lambda s: mock_sct
            mock_mss.return_value.__exit__ = MagicMock(return_value=False)

            with patch("mss.tools.to_png"):
                with patch("vision.capture.screen.CAPTURE_DIR", tmp_path):
                    from vision.capture.screen import capture_screen
                    result = capture_screen(prefix="test")
                    assert isinstance(result, Path)

    def test_capture_before_after_returns_callable(self, tmp_path):
        with patch("vision.capture.screen.capture_screen") as mock_capture:
            mock_capture.return_value = tmp_path / "screen.png"
            from vision.capture.screen import capture_before_after
            before, capture_after = capture_before_after("test_action")
            assert callable(capture_after)


class TestVLMClient:
    def test_describe_screen_calls_api(self, tmp_path):
        fake_image = tmp_path / "test.png"
        fake_image.write_bytes(b"\x89PNG\r\n" + b"\x00" * 100)

        mock_response = MagicMock()
        mock_response.choices[0].message.content = json.dumps({
            "screen_description": "A web form",
            "interactive_elements": [],
            "current_state": "Form is visible",
            "suggested_next_action": "Fill the first field",
        })

        with patch("vision.vlm.client._client") as mock_client:
            with patch("vision.vlm.client._get_trace", return_value=MagicMock(generation=MagicMock(return_value=MagicMock(end=MagicMock())))):
                with patch("vision.vlm.client.encode_image", return_value="ZmFrZQ=="):
                    mock_client.chat.completions.create.return_value = mock_response
                    from vision.vlm.client import describe_screen
                    result = describe_screen(fake_image)
                    assert "screen_description" in result
                    mock_client.chat.completions.create.assert_called_once()

    def test_locate_element_returns_json(self, tmp_path):
        fake_image = tmp_path / "test.png"
        fake_image.write_bytes(b"\x89PNG\r\n" + b"\x00" * 100)

        mock_response = MagicMock()
        mock_response.choices[0].message.content = json.dumps({
            "found": True,
            "element_description": "Submit button",
            "approximate_location": "bottom-right",
            "confidence": 0.92,
            "visual_cues": "Blue button labeled Submit",
        })

        with patch("vision.vlm.client._client") as mock_client:
            with patch("vision.vlm.client._get_trace", return_value=MagicMock(generation=MagicMock(return_value=MagicMock(end=MagicMock())))):
                with patch("vision.vlm.client.encode_image", return_value="ZmFrZQ=="):
                    mock_client.chat.completions.create.return_value = mock_response
                    from vision.vlm.client import locate_element
                    result = locate_element(fake_image, "Submit button")
                    parsed = json.loads(result)
                    assert parsed["found"] is True
                    assert parsed["confidence"] == 0.92


class TestAgentGraph:
    def test_initial_state_structure(self):
        from agent.graph import AgentState
        state: AgentState = {
            "task": "Fill the form",
            "plan": [],
            "current_step": 0,
            "screen_state": None,
            "last_action": None,
            "last_verification": None,
            "retry_count": 0,
            "status": "running",
            "history": [],
            "error": None,
        }
        assert state["task"] == "Fill the form"
        assert state["status"] == "running"
        assert state["retry_count"] == 0

    def test_route_after_verify_success(self):
        from agent.graph import route_after_verify
        state = {
            "status": "running",
            "last_verification": {"success": True, "confidence": 0.9},
            "plan": [{"step": 1}],
            "current_step": 0,
        }
        assert route_after_verify(state) == "advance"

    def test_route_after_verify_low_confidence(self):
        from agent.graph import route_after_verify
        state = {
            "status": "running",
            "last_verification": {"success": True, "confidence": 0.5},
            "plan": [{"step": 1}],
            "current_step": 0,
        }
        assert route_after_verify(state) == "handle_error"

    def test_route_after_verify_completed(self):
        from agent.graph import route_after_verify
        from langgraph.graph import END
        state = {
            "status": "completed",
            "last_verification": {"success": True, "confidence": 0.95},
        }
        assert route_after_verify(state) == END
