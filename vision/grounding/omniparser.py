"""
OmniParser v2 — GUI element grounding.
Parses a screenshot into a list of interactable elements with bounding boxes.
"""
import json
from pathlib import Path
from typing import List, Optional
from dataclasses import dataclass

from PIL import Image


@dataclass
class UIElement:
    label: str               # Text visible on the element
    element_type: str        # button | input | dropdown | link | checkbox | text
    x: int                   # Center X pixel coordinate
    y: int                   # Center Y pixel coordinate
    width: int
    height: int
    confidence: float
    bbox: tuple              # (x1, y1, x2, y2)

    def center(self) -> tuple[int, int]:
        return (self.x, self.y)

    def is_clickable(self) -> bool:
        return self.element_type in ("button", "link", "checkbox", "dropdown", "menu_item")


class OmniParser:
    """
    Wraps OmniParser v2 (Microsoft) for GUI element detection.
    Falls back to VLM-only grounding if OmniParser is not available.
    """

    def __init__(self, model_path: Optional[str] = None):
        self._model = None
        self._available = False
        self._try_load(model_path)

    def _try_load(self, model_path: Optional[str]):
        try:
            # OmniParser v2 — install from:
            # https://github.com/microsoft/OmniParser
            from omniparser import OmniParserModel  # type: ignore
            self._model = OmniParserModel(model_path or "microsoft/OmniParser-v2")
            self._available = True
            print("OmniParser v2 loaded")
        except ImportError:
            print("OmniParser not installed — using VLM grounding fallback")
            self._available = False

    def parse(self, screenshot_path: str | Path) -> List[UIElement]:
        """
        Parse screenshot into UI elements with coordinates.

        Returns list of UIElement with pixel-accurate bounding boxes.
        """
        if self._available:
            return self._parse_omniparser(screenshot_path)
        return self._parse_vlm_fallback(screenshot_path)

    def _parse_omniparser(self, screenshot_path: str | Path) -> List[UIElement]:
        image = Image.open(screenshot_path)
        raw = self._model.parse(image)
        elements = []
        for item in raw.get("elements", []):
            x1, y1, x2, y2 = item["bbox"]
            cx = (x1 + x2) // 2
            cy = (y1 + y2) // 2
            elements.append(UIElement(
                label=item.get("text", ""),
                element_type=item.get("type", "unknown"),
                x=cx, y=cy,
                width=x2 - x1,
                height=y2 - y1,
                confidence=item.get("confidence", 1.0),
                bbox=(x1, y1, x2, y2),
            ))
        return elements

    def _parse_vlm_fallback(self, screenshot_path: str | Path) -> List[UIElement]:
        """Use VLM to estimate element positions when OmniParser unavailable."""
        from vision.vlm.client import analyze_screen
        prompt = """List all interactive UI elements visible on screen.
For each element, estimate its position as a fraction of screen dimensions (0.0 to 1.0).

Respond ONLY in JSON:
{
  "elements": [
    {
      "label": "Submit",
      "type": "button",
      "x_ratio": 0.75,
      "y_ratio": 0.85,
      "width_ratio": 0.1,
      "height_ratio": 0.04,
      "confidence": 0.9
    }
  ]
}"""
        raw = analyze_screen(screenshot_path, prompt, trace_name="vlm_grounding")
        image = Image.open(screenshot_path)
        w, h = image.size

        try:
            data = json.loads(raw)
        except json.JSONDecodeError:
            return []

        elements = []
        for item in data.get("elements", []):
            cx = int(item["x_ratio"] * w)
            cy = int(item["y_ratio"] * h)
            ew = int(item.get("width_ratio", 0.1) * w)
            eh = int(item.get("height_ratio", 0.04) * h)
            elements.append(UIElement(
                label=item.get("label", ""),
                element_type=item.get("type", "unknown"),
                x=cx, y=cy,
                width=ew, height=eh,
                confidence=item.get("confidence", 0.7),
                bbox=(cx - ew // 2, cy - eh // 2, cx + ew // 2, cy + eh // 2),
            ))
        return elements

    def find_element_at(
        self,
        screenshot_path: str | Path,
        click_x: int,
        click_y: int,
    ) -> Optional[UIElement]:
        """
        Retourne l'élément dont la bounding box contient (click_x, click_y).
        Match mathématique pur — aucun appel IA.
        Retourne None si aucune bbox ne contient les coordonnées.
        """
        elements = self.parse(screenshot_path)
        for el in elements:
            x1, y1, x2, y2 = el.bbox
            if x1 <= click_x <= x2 and y1 <= click_y <= y2:
                return el
        return None

    def find_element(
        self,
        screenshot_path: str | Path,
        description: str,
    ) -> Optional[UIElement]:
        """
        Find the best matching element for a semantic description.
        Example: find_element(path, "Submit button")
        """
        elements = self.parse(screenshot_path)
        if not elements:
            return None

        desc_lower = description.lower()
        best = None
        best_score = 0.0

        for el in elements:
            score = 0.0
            label_lower = el.label.lower()

            # Exact label match
            if desc_lower in label_lower or label_lower in desc_lower:
                score += 0.6

            # Type match
            for t in ("button", "input", "link", "dropdown", "checkbox"):
                if t in desc_lower and el.element_type == t:
                    score += 0.3

            score *= el.confidence

            if score > best_score:
                best_score = score
                best = el

        return best if best_score > 0.2 else None


# Singleton
_parser = None


def get_parser() -> OmniParser:
    global _parser
    if _parser is None:
        _parser = OmniParser()
    return _parser
