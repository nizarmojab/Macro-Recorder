"""
Replay Engine — Replays an annotated workflow directly from recorded events.

Fast path: keyboard/type/scroll execute without any VLM call.
Only click events use a single VLM locate call to find the element by label.
"""
import json
import time
from pathlib import Path

RECORDINGS_DIR = Path("data/recordings")

# Map ASCII control characters (from pynput Ctrl+key) back to key names
_CTRL_CHAR_MAP = {chr(i): chr(ord('a') + i - 1) for i in range(1, 27)}


def replay_workflow(workflow_name: str, semantic: bool = False) -> dict:
    """
    Replay a recorded + annotated workflow by executing events directly.
    Much faster than the full agent loop — no observe/verify overhead.
    """
    path = RECORDINGS_DIR / f"{workflow_name}.json"
    if not path.exists():
        raise FileNotFoundError(f"Workflow not found: {path}")

    with open(path) as f:
        recording = json.load(f)

    if not recording.get("annotated"):
        raise ValueError(
            f"Workflow '{workflow_name}' has not been annotated yet. "
            "Run the recorder's annotate() method first."
        )

    events = recording.get("events", [])
    print(f"\nReplaying workflow: {workflow_name}")
    print(f"Description: {recording.get('task_description', '')}")
    print(f"Steps: {len(events)}\n")

    results = []
    for i, event in enumerate(events):
        print(f"[{i+1}/{len(events)}] {event['event_type']} — {event.get('label', '')}")
        try:
            result = _replay_event(event, semantic=semantic)
            print(f"  ✓ {result}")
            results.append({"step": i, "success": True, "result": result})
        except Exception as e:
            print(f"  ✗ {e}")
            results.append({"step": i, "success": False, "error": str(e)})
        time.sleep(0.2)

    success = sum(1 for r in results if r["success"])
    print(f"\nDone: {success}/{len(events)} steps succeeded.")
    return {"status": "completed", "results": results}


def _replay_event(event: dict, semantic: bool = False) -> dict:
    """Execute a single recorded event."""
    from agent.actions.executor import execute_action

    etype = event["event_type"]

    if etype == "type":
        text = event.get("text") or ""
        return execute_action("type", "", value=text)

    elif etype in ("key", "shortcut"):
        key = event.get("key") or ""
        key = _normalize_key(key)
        return execute_action("key", "", value=key)

    elif etype == "click":
        label = event.get("semantic_label") or event.get("label") or ""
        x, y = event.get("raw_x"), event.get("raw_y")
        recorded = {"found": True, "x": x, "y": y} if x and y else None

        if semantic:
            # VLM finds element on current screen, falls back to recorded coords
            location_data = _locate_by_vlm(label) or recorded
        else:
            location_data = recorded

        return execute_action("click", label, location_data=location_data)

    elif etype == "scroll":
        from vision.capture.screen import capture_screen
        screenshot = str(capture_screen(prefix="replay_scroll"))
        amount = event.get("scroll_dy", 3)
        return execute_action("scroll", "current position", value=str(amount),
                              screenshot_path=screenshot)

    elif etype == "double_click":
        import pyautogui
        x, y = event.get("raw_x", 0), event.get("raw_y", 0)
        pyautogui.doubleClick(x, y)
        return {"action": "double_click", "x": x, "y": y}

    else:
        return {"action": "skip", "reason": f"unknown event type: {etype}"}


def _locate_by_vlm(label: str) -> dict:
    """Use VLM to find an element on the current screen by its semantic label."""
    import json
    import re
    from vision.capture.screen import capture_screen
    from vision.vlm.client import locate_element

    screenshot = str(capture_screen(prefix="locate"))
    raw = locate_element(screenshot, label)

    # Strip markdown code blocks if present
    text = raw.strip()
    match = re.search(r"```(?:json)?\s*([\s\S]*?)```", text)
    if match:
        text = match.group(1).strip()
    try:
        data = json.loads(text)
    except Exception:
        data = {}

    if data.get("found") and data.get("x_pct") is not None:
        import pyautogui
        sw, sh = pyautogui.size()
        x = int(data["x_pct"] * sw)
        y = int(data["y_pct"] * sh)
        print(f"    [VLM] Found '{label}' at ({x}, {y}) confidence={data.get('confidence', '?')}")
        return {"found": True, "x": x, "y": y}

    print(f"    [VLM] Could not find '{label}', using recorded coords")
    return None


def _normalize_key(key: str) -> str:
    """
    Normalize key names from pynput format to pyautogui format.
    Also fixes ctrl+<control_char> back to ctrl+<letter>.
    """
    # Fix ctrl+<control_char> e.g. ctrl+\x13 → ctrl+s
    if "+" in key:
        parts = key.split("+")
        normalized = []
        for part in parts:
            if len(part) == 1 and part in _CTRL_CHAR_MAP:
                normalized.append(_CTRL_CHAR_MAP[part])
            else:
                normalized.append(part)
        key = "+".join(normalized)

    # Strip "Key." prefix from pynput names
    key = key.replace("Key.", "").lower()

    # Map pynput names to pyautogui names
    _MAP = {
        "backspace": "backspace",
        "caps_lock": "capslock",
        "enter": "enter",
        "esc": "escape",
        "tab": "tab",
        "space": "space",
        "delete": "delete",
        "up": "up", "down": "down", "left": "left", "right": "right",
        "f1": "f1", "f2": "f2", "f3": "f3", "f4": "f4",
        "f5": "f5", "f6": "f6", "f7": "f7", "f8": "f8",
        "f9": "f9", "f10": "f10", "f11": "f11", "f12": "f12",
    }
    return _MAP.get(key, key)


def list_workflows() -> list[dict]:
    """List all available recorded workflows."""
    workflows = []
    for f in RECORDINGS_DIR.glob("*.json"):
        with open(f) as fp:
            data = json.load(fp)
        workflows.append({
            "name": data["workflow_name"],
            "description": data.get("task_description", ""),
            "annotated": data.get("annotated", False),
            "event_count": len(data.get("events", [])),
            "recorded_at": data.get("started_at", ""),
        })
    return workflows
