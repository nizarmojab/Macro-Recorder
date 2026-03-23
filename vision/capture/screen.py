"""
Screen Capture — Fast screenshot capture using mss.
All captures go through this module.
"""
import time
from datetime import datetime
from pathlib import Path
from typing import Optional

import mss
import mss.tools
from PIL import Image


CAPTURE_DIR = Path("data/captures")
CAPTURE_DIR.mkdir(parents=True, exist_ok=True)


def capture_screen(
    monitor: int = 1,
    region: Optional[dict] = None,
    save: bool = True,
    prefix: str = "screen",
) -> Path:
    """
    Capture the screen or a region.

    Args:
        monitor: Monitor index (1 = primary)
        region: Optional {"top": y, "left": x, "width": w, "height": h}
        save: Whether to save to disk
        prefix: Filename prefix

    Returns:
        Path to the saved screenshot
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    output_path = CAPTURE_DIR / f"{prefix}_{timestamp}.png"

    with mss.mss() as sct:
        target = region if region else sct.monitors[monitor]
        screenshot = sct.grab(target)
        if save:
            mss.tools.to_png(screenshot.rgb, screenshot.size, output=str(output_path))

    return output_path


def capture_before_after(action_name: str) -> tuple[Path, callable]:
    """
    Capture before screenshot, return path + a callable to capture after.

    Usage:
        before, capture_after = capture_before_after("click_submit")
        # ... perform action ...
        after = capture_after()
    """
    before = capture_screen(prefix=f"{action_name}_before")

    def capture_after() -> Path:
        return capture_screen(prefix=f"{action_name}_after")

    return before, capture_after


def capture_region(x: int, y: int, width: int, height: int) -> Path:
    """Capture a specific region of the screen."""
    region = {"top": y, "left": x, "width": width, "height": height}
    return capture_screen(region=region, prefix="region")


def wait_for_screen_change(
    baseline_path: Path,
    timeout: float = 10.0,
    interval: float = 0.5,
    threshold: float = 0.02,
) -> Optional[Path]:
    """
    Poll the screen until it changes from the baseline.

    Args:
        baseline_path: Screenshot to compare against
        timeout: Max seconds to wait
        interval: Seconds between polls
        threshold: Minimum fraction of pixels that must change

    Returns:
        Path to changed screenshot, or None if timeout
    """
    baseline = Image.open(baseline_path).convert("RGB")
    baseline_pixels = list(baseline.getdata())
    total_pixels = len(baseline_pixels)

    start = time.time()
    while time.time() - start < timeout:
        time.sleep(interval)
        current_path = capture_screen(prefix="change_detection")
        current = Image.open(current_path).convert("RGB")
        current_pixels = list(current.getdata())

        changed = sum(
            1 for a, b in zip(baseline_pixels, current_pixels) if a != b
        )
        if changed / total_pixels > threshold:
            return current_path

    return None
