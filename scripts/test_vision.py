"""
scripts/test_vision.py — Quick sanity check for the VLM + capture pipeline.
Run this first to verify everything is working before starting development.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from dotenv import load_dotenv
load_dotenv()

from vision.capture.screen import capture_screen
from vision.vlm.client import describe_screen, locate_element


def test_capture():
    print("1. Testing screen capture...")
    path = capture_screen(prefix="test")
    assert path.exists(), f"Screenshot not found: {path}"
    print(f"   ✓ Screenshot saved: {path}")
    return path


def test_vlm(screenshot_path):
    print("2. Testing VLM (screen description)...")
    result = describe_screen(screenshot_path)
    print(f"   ✓ VLM responded ({len(result)} chars)")
    print(f"   Preview: {result[:200]}...")
    return result


def test_grounding(screenshot_path):
    print("3. Testing element grounding...")
    result = locate_element(screenshot_path, "any button or clickable element visible on screen")
    print(f"   ✓ Grounding responded")
    print(f"   Preview: {result[:200]}...")


def main():
    print("\n=== Vision Pipeline Test ===\n")
    try:
        path = test_capture()
        test_vlm(path)
        test_grounding(path)
        print("\n✓ All tests passed — vision pipeline is working\n")
    except Exception as e:
        print(f"\n✗ Test failed: {e}\n")
        print("Check that vLLM is running: vllm serve Qwen/Qwen2.5-VL-7B-Instruct --host 0.0.0.0 --port 8000")
        sys.exit(1)


if __name__ == "__main__":
    main()
