"""
VLM Client — Local Qwen2.5-VL via vLLM
All vision calls go through this module. Never call the API directly elsewhere.
"""
import base64
import os
import time
from pathlib import Path
from typing import Optional

import openai
from dotenv import load_dotenv

load_dotenv()

VLM_BASE_URL = os.getenv("VLM_BASE_URL", "http://localhost:8000/v1")
VLM_MODEL = os.getenv("VLM_MODEL", "Qwen/Qwen2.5-VL-7B-Instruct")
VLM_API_KEY = os.getenv("TOGETHER_API_KEY") or os.getenv("GEMINI_API_KEY") or os.getenv("OPENAI_API_KEY") or "local"

_client = openai.OpenAI(base_url=VLM_BASE_URL, api_key=VLM_API_KEY)


class _NoopGeneration:
    def end(self, **kwargs):
        pass


class _NoopTrace:
    def generation(self, **kwargs):
        return _NoopGeneration()


def _get_trace(name: str):
    """Return a Langfuse trace if available, else a no-op."""
    try:
        from langfuse import Langfuse
        lf = Langfuse()
        return lf.trace(name=name)
    except Exception:
        return _NoopTrace()


def encode_image(image_path: str | Path, max_width: int = 480) -> str:
    """Encode image to base64 string, resizing to reduce token cost."""
    from PIL import Image
    import io
    img = Image.open(image_path)
    if img.width > max_width:
        ratio = max_width / img.width
        new_size = (max_width, int(img.height * ratio))
        img = img.resize(new_size, Image.LANCZOS)
    buffer = io.BytesIO()
    img.save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode("utf-8")


def analyze_screen(
    screenshot_path: str | Path,
    prompt: str,
    trace_name: str = "analyze_screen",
    region: Optional[dict] = None,
    max_tokens: int = 1024,
    json_mode: bool = False,
) -> str:
    """
    Send a screenshot to the local VLM and get a response.

    Args:
        screenshot_path: Path to the screenshot file
        prompt: What to ask the VLM about the screen
        trace_name: Name for Langfuse tracing
        region: Optional dict with x, y, width, height to crop before sending

    Returns:
        VLM response string
    """
    img_b64 = encode_image(screenshot_path)

    trace = _get_trace(trace_name)
    generation = trace.generation(
        name="vlm_call",
        model=VLM_MODEL,
        input={"prompt": prompt, "has_image": True},
    )

    for attempt in range(4):
        try:
            kwargs = dict(
                model=VLM_MODEL,
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/png;base64,{img_b64}"
                                },
                            },
                            {"type": "text", "text": prompt},
                        ],
                    }
                ],
                max_tokens=max_tokens,
                temperature=0.1,
            )
            if json_mode:
                kwargs["response_format"] = {"type": "json_object"}
            response = _client.chat.completions.create(**kwargs)
            result = response.choices[0].message.content
            generation.end(output=result)
            return result

        except openai.AuthenticationError as e:
            generation.end(level="ERROR", status_message=str(e))
            raise RuntimeError(f"Clé API invalide: {e}") from e

        except openai.RateLimitError as e:
            wait = 2 ** attempt
            print(f"  [vlm] Rate limit ({e}), retry in {wait}s...")
            time.sleep(wait)

        except Exception as e:
            generation.end(level="ERROR", status_message=str(e))
            raise

    generation.end(level="ERROR", status_message="Rate limit retries exhausted")
    raise RuntimeError("VLM rate limit retries exhausted")


def describe_screen(screenshot_path: str | Path) -> str:
    """Get a structured description of what is on screen."""
    prompt = """Analyze this screenshot and respond in JSON with this exact structure:
{
  "screen_description": "What is visible on screen",
  "application": "Name of the application if identifiable",
  "interactive_elements": [
    {"type": "button|input|dropdown|link", "label": "visible text", "location": "description of where it is"}
  ],
  "current_state": "What state the application is in",
  "suggested_next_action": "What a user would logically do next"
}
Respond with JSON only."""
    return analyze_screen(screenshot_path, prompt, trace_name="describe_screen")


def locate_element(screenshot_path: str | Path, element_description: str) -> str:
    """Ask the VLM to locate a specific element on screen, returning coordinates as % of screen."""
    prompt = f"""Locate this element on the screen: "{element_description}"

Respond in JSON:
{{
  "found": true,
  "element_description": "What you found",
  "x_pct": 0.5,
  "y_pct": 0.8,
  "confidence": 0.95,
  "visual_cues": "What makes this element identifiable"
}}

x_pct and y_pct are the center of the element as a fraction of screen width/height (0.0 to 1.0)."""
    return analyze_screen(screenshot_path, prompt, trace_name="locate_element")


def verify_action(
    before_path: str | Path,
    after_path: str | Path,
    expected_outcome: str,
) -> str:
    """Compare before/after screenshots to verify an action succeeded."""
    before_b64 = encode_image(before_path)
    after_b64 = encode_image(after_path)

    trace = _get_trace("verify_action")
    generation = trace.generation(
        name="vlm_verify",
        model=VLM_MODEL,
        input={"expected": expected_outcome},
    )

    for attempt in range(4):
        try:
            response = _client.chat.completions.create(
                model=VLM_MODEL,
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": "BEFORE the action:"},
                            {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{before_b64}"}},
                            {"type": "text", "text": "AFTER the action:"},
                            {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{after_b64}"}},
                            {"type": "text", "text": f"""Expected outcome: {expected_outcome}

Did the action succeed? Respond in JSON:
{{
  "success": true,
  "confidence": 0.92,
  "changes_detected": ["list of visible changes"],
  "assessment": "explanation",
  "needs_human_review": false
}}"""},
                        ],
                    }
                ],
                max_tokens=512,
                temperature=0.1,
            )
            result = response.choices[0].message.content
            generation.end(output=result)
            return result

        except openai.RateLimitError as e:
            wait = 2 ** attempt
            print(f"  [vlm] Rate limit hit, retrying in {wait}s...")
            time.sleep(wait)

        except Exception as e:
            generation.end(level="ERROR", status_message=str(e))
            raise

    generation.end(level="ERROR", status_message="Rate limit retries exhausted")
    raise RuntimeError("VLM rate limit retries exhausted")
