"""
SmartRecorder — 3 threads simultanés :
  Thread 1 (pynput)        : capture clics souris + touches clavier
  Thread 2 (CursorOverlay) : fenêtre tkinter transparente fullscreen,
                             cercle rouge sur le curseur à 60fps,
                             animation ripple sur chaque clic (400ms)
  Thread 3 (mss)           : screenshot après chaque clic, overlay visible

Annotation en temps réel via OmniParser (match mathématique bbox).
Fallback VLM : crop 200×200 autour du clic si OmniParser non dispo.
"""
import json
import time
import ctypes
import threading
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import List, Optional

from pynput import mouse, keyboard
from vision.capture.screen import capture_screen


RECORDINGS_DIR = Path("data/recordings")
RECORDINGS_DIR.mkdir(parents=True, exist_ok=True)


# ── Data model ────────────────────────────────────────────────────────────────

@dataclass
class RecordedEvent:
    timestamp: float
    event_type: str         # click | double_click | key | type | scroll
    raw_x: Optional[int]    # coordonnées brutes (debug uniquement)
    raw_y: Optional[int]
    key: Optional[str]
    text: Optional[str]
    screenshot_path: str
    label: str = ""         # label OmniParser / VLM
    elem_type: str = ""     # button | input | dropdown | link | …
    confidence: float = 0.0
    annotated: bool = False
    # Champs legacy conservés pour le replay engine
    semantic_label: str = ""
    intent: str = ""


@dataclass
class Recording:
    workflow_name: str
    task_description: str
    started_at: str
    events: List[RecordedEvent] = field(default_factory=list)
    annotated: bool = False


# ── Overlay tkinter ───────────────────────────────────────────────────────────

class CursorOverlay:
    """
    Fenêtre transparente fullscreen qui affiche :
    - Un cercle rouge + croix centré sur le curseur (60 fps)
    - Un ripple animé à chaque clic (cercle qui grandit et disparaît en 400 ms)
    Click-through via Windows API (WS_EX_TRANSPARENT | WS_EX_LAYERED).
    """

    def __init__(self):
        self._root = None
        self._running = False
        self._ripples: list = []          # [(x, y, t0), …]
        self._pending_ripple: Optional[tuple] = None
        self._lock = threading.Lock()
        self._ready = threading.Event()

    def start(self):
        self._running = True
        t = threading.Thread(target=self._run, daemon=True)
        t.start()
        self._ready.wait(timeout=2.0)     # attendre que tkinter soit prêt

    def hide(self):
        if self._root:
            try:
                self._root.after(0, self._root.withdraw)
            except Exception:
                pass

    def show(self):
        if self._root:
            try:
                self._root.after(0, self._root.deiconify)
            except Exception:
                pass

    def trigger_ripple(self, x: int, y: int):
        with self._lock:
            self._pending_ripple = (x, y)

    def stop(self):
        self._running = False
        if self._root:
            try:
                self._root.after(0, self._root.quit)
            except Exception:
                pass

    def _run(self):
        import tkinter as tk

        root = tk.Tk()
        self._root = root

        sw = root.winfo_screenwidth()
        sh = root.winfo_screenheight()

        root.overrideredirect(True)
        root.attributes("-topmost", True)
        root.geometry(f"{sw}x{sh}+0+0")

        root.configure(bg="black")
        root.wm_attributes("-alpha", 0.01)

        canvas = tk.Canvas(root, width=sw, height=sh,
                           bg="black", highlightthickness=0)
        canvas.pack()

        def apply_click_through():
            try:
                hwnd = root.winfo_id()
                GWL_EXSTYLE = -20
                WS_EX_LAYERED = 0x00080000
                WS_EX_TRANSPARENT = 0x00000020
                style = ctypes.windll.user32.GetWindowLongW(hwnd, GWL_EXSTYLE)
                ctypes.windll.user32.SetWindowLongW(
                    hwnd, GWL_EXSTYLE, style | WS_EX_LAYERED | WS_EX_TRANSPARENT
                )
            except Exception:
                pass
            self._ready.set()

        root.after(200, apply_click_through)

        def update():
            if not self._running:
                root.quit()
                return

            canvas.delete("overlay")

            # Position du curseur
            try:
                mx = root.winfo_pointerx()
                my = root.winfo_pointery()
            except Exception:
                root.after(16, update)
                return

            r = 14
            canvas.create_oval(mx - r, my - r, mx + r, my + r,
                                outline="red", width=2, tags="overlay")
            canvas.create_line(mx - r, my, mx + r, my,
                                fill="red", width=1, tags="overlay")
            canvas.create_line(mx, my - r, mx, my + r,
                                fill="red", width=1, tags="overlay")

            # Consommer le ripple en attente
            with self._lock:
                pending = self._pending_ripple
                self._pending_ripple = None
            if pending:
                self._ripples.append((pending[0], pending[1], time.time()))

            # Dessiner et mettre à jour les ripples
            now = time.time()
            alive = []
            for (rx, ry, t0) in self._ripples:
                elapsed = now - t0
                if elapsed >= 0.4:
                    continue
                alive.append((rx, ry, t0))
                progress = elapsed / 0.4
                radius = int(8 + 50 * progress)
                stroke = max(1, int(3 * (1.0 - progress)))
                canvas.create_oval(
                    rx - radius, ry - radius, rx + radius, ry + radius,
                    outline="red", width=stroke, tags="overlay",
                )
            self._ripples = alive

            root.after(16, update)

        root.after(16, update)
        root.mainloop()


# ── Recorder principal ────────────────────────────────────────────────────────

class ActionRecorder:
    """
    Enregistre les actions humaines avec feedback visuel et annotation sémantique
    en temps réel.
    """

    def __init__(self, workflow_name: str, task_description: str = ""):
        self.workflow_name = workflow_name
        self.task_description = task_description
        self.recording = Recording(
            workflow_name=workflow_name,
            task_description=task_description,
            started_at=datetime.now().isoformat(),
        )
        self._running = False
        self._text_buffer = ""
        self._mouse_listener = None
        self._keyboard_listener = None
        self._modifiers: set = set()
        self._last_click_time: float = 0
        self._last_click_pos: tuple = (0, 0)
        from concurrent.futures import ThreadPoolExecutor
        self._executor = ThreadPoolExecutor(max_workers=3)

    def start(self):
        """Lance les listeners pynput."""
        self._running = True
        print(f"Recording '{self.workflow_name}'... Press Ctrl+Shift+S to stop.")

        self._mouse_listener = mouse.Listener(
            on_click=self._on_click,
            on_scroll=self._on_scroll,
        )
        self._keyboard_listener = keyboard.Listener(
            on_press=self._on_key_press,
            on_release=self._on_key_release,
        )
        self._mouse_listener.start()
        self._keyboard_listener.start()

    def stop(self) -> Path:
        """Arrête les listeners et sauvegarde le JSON."""
        self._running = False
        if self._mouse_listener:
            self._mouse_listener.stop()
        if self._keyboard_listener:
            self._keyboard_listener.stop()
        # Attendre que toutes les annotations en cours se terminent
        self._executor.shutdown(wait=True)
        return self._save()

    # ── Handlers pynput ───────────────────────────────────────────────────────

    def _on_click(self, x, y, button, pressed):
        if not self._running or not pressed:
            return

        # Flusher le texte en attente avant le clic
        if self._text_buffer:
            self._flush_text_buffer()

        # Détecter double-clic (2 clics < 400ms au même endroit ±5px)
        now = time.time()
        dx = abs(x - self._last_click_pos[0])
        dy = abs(y - self._last_click_pos[1])
        is_double = (now - self._last_click_time < 0.4 and dx < 5 and dy < 5)
        self._last_click_time = now
        self._last_click_pos = (x, y)

        event_type = "double_click" if is_double else "click"
        event = RecordedEvent(
            timestamp=now,
            event_type=event_type,
            raw_x=x,
            raw_y=y,
            key=str(button),
            text=None,
            screenshot_path="",  # rempli en arrière-plan
        )
        self.recording.events.append(event)
        # Screenshot + annotation entièrement en arrière-plan
        self._executor.submit(self._capture_and_annotate_click, event, x, y)

    def _on_scroll(self, x, y, dx, dy):
        if not self._running:
            return
        event = RecordedEvent(
            timestamp=time.time(),
            event_type="scroll",
            raw_x=x, raw_y=y,
            key=None,
            text=f"dx={dx},dy={dy}",
            screenshot_path="",
            label="scroll",
            elem_type="scroll",
            confidence=1.0,
            annotated=True,
            semantic_label="Scroll",
            intent=f"Scroll {'down' if dy < 0 else 'up'}",
        )
        self.recording.events.append(event)

    _MODIFIERS = {
        keyboard.Key.ctrl, keyboard.Key.ctrl_l, keyboard.Key.ctrl_r,
        keyboard.Key.alt, keyboard.Key.alt_l, keyboard.Key.alt_r,
        keyboard.Key.shift, keyboard.Key.shift_l, keyboard.Key.shift_r,
        keyboard.Key.cmd, keyboard.Key.cmd_l, keyboard.Key.cmd_r,
    }

    def _on_key_press(self, key):
        if not self._running:
            return

        # Suivre les modificateurs
        if key in self._MODIFIERS:
            self._modifiers.add(key)
            return

        # Si un modificateur est actif → c'est un raccourci
        if self._modifiers:
            if self._text_buffer:
                self._flush_text_buffer()
            self._record_shortcut(key)
            return

        # Caractère normal → buffer
        try:
            char = key.char
            if char:
                self._text_buffer += char
                return
        except AttributeError:
            pass

        # Touche spéciale (Enter, Tab, flèches...) → flush + enregistrer
        if self._text_buffer:
            self._flush_text_buffer()

        screenshot = capture_screen(prefix="key")
        key_str = str(key)
        event = RecordedEvent(
            timestamp=time.time(),
            event_type="key",
            raw_x=None, raw_y=None,
            key=key_str,
            text=None,
            screenshot_path=str(screenshot),
            label=key_str,
            elem_type="key",
            confidence=1.0,
            annotated=True,
            semantic_label=f"Key: {key_str}",
            intent=f"Press {key_str}",
        )
        self.recording.events.append(event)
        print(f"  ✓ Key → {key_str}")

    def _on_key_release(self, key):
        self._modifiers.discard(key)
        if key == keyboard.Key.esc:
            self.stop()

    def _record_shortcut(self, key):
        """Enregistre un raccourci clavier (Ctrl+C, Win+R, etc.)"""
        mod_names = []
        if any(k in self._modifiers for k in (keyboard.Key.ctrl, keyboard.Key.ctrl_l, keyboard.Key.ctrl_r)):
            mod_names.append("ctrl")
        if any(k in self._modifiers for k in (keyboard.Key.alt, keyboard.Key.alt_l, keyboard.Key.alt_r)):
            mod_names.append("alt")
        if any(k in self._modifiers for k in (keyboard.Key.shift, keyboard.Key.shift_l, keyboard.Key.shift_r)):
            mod_names.append("shift")
        if any(k in self._modifiers for k in (keyboard.Key.cmd, keyboard.Key.cmd_l, keyboard.Key.cmd_r)):
            mod_names.append("win")

        try:
            char = key.char or str(key)
            # Ctrl+lettre → pynput donne un caractère de contrôle (ex: Ctrl+S = \x13)
            # On remonte à la lettre originale: \x01=a, \x02=b, ..., \x1a=z
            if char and len(char) == 1 and ord(char) < 32:
                char = chr(ord(char) + ord('a') - 1)
            key_name = char
        except AttributeError:
            key_name = str(key).replace("Key.", "")

        shortcut = "+".join(mod_names + [key_name])
        screenshot = capture_screen(prefix="shortcut")
        event = RecordedEvent(
            timestamp=time.time(),
            event_type="shortcut",
            raw_x=None, raw_y=None,
            key=shortcut,
            text=None,
            screenshot_path=str(screenshot),
            label=shortcut,
            elem_type="shortcut",
            confidence=1.0,
            annotated=True,
            semantic_label=f"Shortcut: {shortcut}",
            intent=f"Press {shortcut}",
        )
        self.recording.events.append(event)
        print(f"  ✓ Shortcut → {shortcut}")

    def _flush_text_buffer(self):
        if not self._text_buffer:
            return
        screenshot = capture_screen(prefix="type")
        text = self._text_buffer
        event = RecordedEvent(
            timestamp=time.time(),
            event_type="type",
            raw_x=None, raw_y=None,
            key=None,
            text=text,
            screenshot_path=str(screenshot),
            label=f'"{text}"',
            elem_type="text_input",
            confidence=1.0,
            annotated=True,
            semantic_label=f'Typed: "{text}"',
            intent=f'Type "{text}"',
        )
        self.recording.events.append(event)
        self._text_buffer = ""

    # ── Annotation en temps réel ──────────────────────────────────────────────

    def _capture_and_annotate_click(self, event: RecordedEvent, x: int, y: int):
        """Prend le screenshot ET annote — tourne entièrement en background."""
        screenshot = capture_screen(prefix="click")
        event.screenshot_path = str(screenshot)
        self._annotate_click_event(event, x, y, screenshot)

    def _annotate_click_event(self, event: RecordedEvent, x: int, y: int,
                               screenshot_path: Path):
        """OmniParser (match bbox mathématique) ou fallback VLM 200×200."""
        from vision.grounding.omniparser import get_parser

        parser = get_parser()
        found = None

        if parser._available:
            found = parser.find_element_at(screenshot_path, x, y)

        if found:
            event.label = found.label
            event.elem_type = found.element_type
            event.confidence = found.confidence
            event.annotated = True
            event.semantic_label = f"{found.element_type.capitalize()}: {found.label}"
            event.intent = f"Click on {found.label}"
            print(f"  ✓ Click → {found.label} ({found.element_type})")
        else:
            self._annotate_with_vlm(event, x, y, screenshot_path)

    def _annotate_with_vlm(self, event: RecordedEvent, x: int, y: int,
                            screenshot_path: Path):
        """Fallback VLM : crop 200×200 centré sur le clic."""
        from PIL import Image, ImageDraw
        from vision.vlm.client import analyze_screen

        try:
            img = Image.open(screenshot_path)
            half = 100
            x1 = max(0, x - half)
            y1 = max(0, y - half)
            x2 = min(img.width, x + half)
            y2 = min(img.height, y + half)
            if x2 <= x1 or y2 <= y1:
                x1, y1, x2, y2 = 0, 0, min(200, img.width), min(200, img.height)
            crop = img.crop((x1, y1, x2, y2))

            draw = ImageDraw.Draw(crop)
            cx, cy, r = x - x1, y - y1, 10
            draw.ellipse([cx - r, cy - r, cx + r, cy + r], outline="red", width=2)
            draw.line([cx - r, cy, cx + r, cy], fill="red", width=1)
            draw.line([cx, cy - r, cx, cy + r], fill="red", width=1)

            crop_path = Path(str(screenshot_path)).with_suffix(".crop.png")
            crop.save(crop_path)

            prompt = (
                "A red circle marks where the user clicked. "
                "What UI element is at that location?\n"
                "Respond ONLY in JSON:\n"
                '{"label":"short name","elem_type":"button|input|link|dropdown|checkbox|menu_item|text|other",'
                '"confidence":0.0,"intent":"The user wanted to..."}'
            )
            raw = analyze_screen(str(crop_path), prompt,
                                 trace_name="annotate_click_vlm")

            clean = raw.strip()
            if clean.startswith("```"):
                parts = clean.split("```")
                clean = parts[1] if len(parts) > 1 else clean
                if clean.startswith("json"):
                    clean = clean[4:]

            data = json.loads(clean.strip())
            event.label = data.get("label", "")
            event.elem_type = data.get("elem_type", "other")
            event.confidence = float(data.get("confidence", 0.5))
            event.annotated = True
            event.semantic_label = (
                f"{event.elem_type.capitalize()}: {event.label}"
            )
            event.intent = data.get("intent", f"Click on {event.label}")
            print(f"  ✓ Click → {event.label} ({event.elem_type}) [VLM]")

        except Exception as e:
            event.label = "unknown"
            event.elem_type = "other"
            event.annotated = False
            print(f"  ✗ Annotation failed ({x}, {y}): {e}")

    # ── Persistance ───────────────────────────────────────────────────────────

    def _save(self) -> Path:
        if self._text_buffer:
            self._flush_text_buffer()
        output_path = RECORDINGS_DIR / f"{self.workflow_name}.json"
        with open(output_path, "w") as f:
            json.dump(
                {
                    "workflow_name": self.recording.workflow_name,
                    "task_description": self.recording.task_description,
                    "started_at": self.recording.started_at,
                    "annotated": True,
                    "events": [asdict(e) for e in self.recording.events],
                },
                f,
                indent=2,
            )
        print(f"Recording saved: {output_path}")
        return output_path

    def annotate(self, recording_path: Path) -> Path:
        """
        Post-annotation pour les enregistrements legacy (sans annotation temps réel).
        Ignore les événements déjà annotés.
        """
        with open(recording_path) as f:
            data = json.load(f)

        pending = [
            e for e in data["events"]
            if e["event_type"] in ("click", "double_click")
            and not e.get("annotated")
        ]

        if not pending:
            print(f"Annotation complete: {recording_path} (already annotated)")
            return Path(recording_path)

        from vision.vlm.client import analyze_screen

        for event in pending:
            marked_path = self._mark_click(
                event["screenshot_path"], event["raw_x"], event["raw_y"]
            )
            prompt = (
                "A RED CIRCLE marks where the user clicked. Respond in JSON:\n"
                '{"semantic_label":"Button: Start","intent":"The user wanted to...",'
                '"element_type":"button|input|link|dropdown|checkbox|menu_item|other"}'
            )
            time.sleep(3)
            raw = analyze_screen(marked_path, prompt,
                                 trace_name="annotate_recording")
            try:
                clean = raw.strip()
                if clean.startswith("```"):
                    parts = clean.split("```")
                    clean = parts[1] if len(parts) > 1 else clean
                    if clean.startswith("json"):
                        clean = clean[4:]
                ann = json.loads(clean.strip())
                event["semantic_label"] = ann.get("semantic_label", "")
                event["intent"] = ann.get("intent", "")
                event["label"] = ann.get("semantic_label", "")
                event["elem_type"] = ann.get("element_type", "other")
                event["annotated"] = True
            except json.JSONDecodeError:
                event["semantic_label"] = "unknown"
                event["intent"] = "unknown"

        data["annotated"] = True
        with open(recording_path, "w") as f:
            json.dump(data, f, indent=2)

        print(f"Annotation complete: {recording_path}")
        return Path(recording_path)

    def _mark_click(self, screenshot_path: str, x: int, y: int) -> Path:
        from PIL import Image, ImageDraw
        img = Image.open(screenshot_path).convert("RGB")
        draw = ImageDraw.Draw(img)
        r = 20
        draw.ellipse([x - r, y - r, x + r, y + r], outline="red", width=4)
        draw.line([x - r, y, x + r, y], fill="red", width=2)
        draw.line([x, y - r, x, y + r], fill="red", width=2)
        marked_path = Path(screenshot_path).with_suffix(".marked.png")
        img.save(marked_path)
        return marked_path
