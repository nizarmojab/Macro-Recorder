"""
Action Executor — Translates semantic actions into system-level interactions.
Chooses the right tool (pywinauto, pyautogui, Playwright) for each context.
"""
import time
from typing import Optional

import pyautogui
from vision.grounding.omniparser import get_parser, UIElement

pyautogui.FAILSAFE = True   # Move mouse to corner to abort
pyautogui.PAUSE = 0.3       # Small pause between actions


def execute_action(
    action_type: str,
    target_description: str,
    location_data: Optional[dict] = None,
    value: Optional[str] = None,
    screenshot_path: Optional[str] = None,
) -> dict:
    """
    Execute an action on screen.

    Args:
        action_type: click | type | scroll | key | wait | drag
        target_description: Semantic description of where to act
        location_data: Optional pre-computed location from grounding agent
        value: Text to type, key to press, or scroll amount
        screenshot_path: Current screenshot for grounding (if needed)

    Returns:
        dict with action result info
    """
    if action_type == "click":
        return _click(target_description, location_data, screenshot_path)
    elif action_type == "type":
        return _type(value or "")
    elif action_type == "key":
        return _key(value or "")
    elif action_type == "scroll":
        return _scroll(target_description, value, location_data, screenshot_path)
    elif action_type == "wait":
        seconds = float(value or "2")
        time.sleep(seconds)
        return {"action": "wait", "duration": seconds}
    elif action_type == "drag":
        return _drag(target_description, value, screenshot_path)
    else:
        raise ValueError(f"Unknown action type: {action_type}")


def _resolve_coordinates(
    target_description: str,
    location_data: Optional[dict],
    screenshot_path: Optional[str],
) -> tuple[int, int]:
    """
    Resolve a semantic target description to pixel coordinates.
    Uses OmniParser grounding — never hardcodes coordinates.
    """
    # If exact coordinates were provided, use them directly
    if location_data and location_data.get("found"):
        x = location_data.get("x")
        y = location_data.get("y")
        if x is not None and y is not None:
            return int(x), int(y)

    # Use OmniParser to find the element
    if screenshot_path:
        from vision.capture.screen import capture_screen
        path = screenshot_path
    else:
        from vision.capture.screen import capture_screen
        path = str(capture_screen(prefix="grounding"))

    parser = get_parser()
    element = parser.find_element(path, target_description)

    if element and element.confidence >= 0.6:
        return element.center()

    raise RuntimeError(
        f"Could not locate element: '{target_description}' "
        f"(best confidence: {element.confidence if element else 0})"
    )


def _click(
    target_description: str,
    location_data: Optional[dict],
    screenshot_path: Optional[str],
) -> dict:
    x, y = _resolve_coordinates(target_description, location_data, screenshot_path)
    _validate_coords(x, y)
    pyautogui.moveTo(x, y, duration=0.3)
    pyautogui.click(x, y)
    return {"action": "click", "x": x, "y": y, "target": target_description}


def _type(text: str) -> dict:
    # Replace dynamic placeholders
    from datetime import date
    import os
    text = text.replace("{TODAY}", date.today().isoformat())
    text = text.replace("{USERNAME}", os.getenv("USERNAME", ""))

    pyautogui.typewrite(text, interval=0.05)
    return {"action": "type", "text": text}


def _key(key_combo: str) -> dict:
    """Press a key or key combination. Example: 'enter', 'ctrl+a', 'tab'"""
    keys = [k.strip() for k in key_combo.lower().split("+")]
    if len(keys) == 1:
        pyautogui.press(keys[0])
    else:
        pyautogui.hotkey(*keys)
    return {"action": "key", "keys": keys}


def _scroll(
    target_description: str,
    value: Optional[str],
    location_data: Optional[dict],
    screenshot_path: Optional[str],
) -> dict:
    try:
        x, y = _resolve_coordinates(target_description, location_data, screenshot_path)
    except RuntimeError:
        # Scroll at current position if target not found
        x, y = pyautogui.position()

    amount = int(value or "3")
    pyautogui.scroll(amount, x=x, y=y)
    return {"action": "scroll", "x": x, "y": y, "amount": amount}


def _drag(
    target_description: str,
    value: Optional[str],
    screenshot_path: Optional[str],
) -> dict:
    x, y = _resolve_coordinates(target_description, None, screenshot_path)
    # value = "to: Drop zone label"
    if value and value.startswith("to:"):
        dest_desc = value[3:].strip()
        dx, dy = _resolve_coordinates(dest_desc, None, screenshot_path)
    else:
        dx, dy = x + 100, y

    pyautogui.moveTo(x, y, duration=0.3)
    pyautogui.dragTo(dx, dy, duration=0.5, button="left")
    return {"action": "drag", "from": (x, y), "to": (dx, dy)}


def _validate_coords(x: int, y: int):
    """Ensure coordinates are within screen bounds."""
    sw, sh = pyautogui.size()
    if not (0 <= x <= sw and 0 <= y <= sh):
        raise ValueError(
            f"Coordinates ({x}, {y}) outside screen bounds ({sw}x{sh})"
        )
