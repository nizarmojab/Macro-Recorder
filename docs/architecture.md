# Architecture — Macro Recorder Agent

## System Overview

```
┌─────────────────────────────────────────────────────────────┐
│                        User Interface                        │
│            CLI (main.py) / API / Web Dashboard              │
└──────────────────────────┬──────────────────────────────────┘
                           │
            ┌──────────────▼──────────────┐
            │        Agent Core           │
            │   LangGraph FSM             │
            │  observe→plan→locate→       │
            │  act→verify→advance         │
            └──┬──────────────────────┬───┘
               │                      │
   ┌───────────▼────────┐  ┌──────────▼──────────┐
   │   Vision Layer     │  │   Execution Layer    │
   │  ─────────────     │  │  ───────────────     │
   │  mss (capture)     │  │  pywinauto (desktop) │
   │  OmniParser v2     │  │  Playwright (web)    │
   │  Grounding DINO    │  │  pyautogui (fallback)│
   │  Qwen2.5-VL (VLM)  │  │  Docker VM (isolat.) │
   └───────────┬────────┘  └──────────┬──────────┘
               │                      │
   ┌───────────▼──────────────────────▼──────────┐
   │              Storage Layer                   │
   │  PostgreSQL  MinIO  Qdrant  Redis            │
   └──────────────────────────────────────────────┘
               │
   ┌───────────▼──────────────────────────────────┐
   │           Observability & Security           │
   │  Langfuse  Prometheus  Grafana  Vault        │
   └──────────────────────────────────────────────┘
```

## Agent Loop (ReAct Pattern)

```
┌─────────────────────────────────────────────────────┐
│                    OBSERVE                          │
│  mss captures screen → VLM describes state         │
└───────────────────────┬─────────────────────────────┘
                        │
┌───────────────────────▼─────────────────────────────┐
│                     PLAN                            │
│  Check Qdrant memory → Generate or reuse plan       │
└───────────────────────┬─────────────────────────────┘
                        │
┌───────────────────────▼─────────────────────────────┐
│                    LOCATE                           │
│  OmniParser finds element → Returns coordinates    │
└───────────────────────┬─────────────────────────────┘
                        │
┌───────────────────────▼─────────────────────────────┐
│                      ACT                            │
│  Log to audit → Execute action → Capture after     │
└───────────────────────┬─────────────────────────────┘
                        │
┌───────────────────────▼─────────────────────────────┐
│                    VERIFY                           │
│  VLM compares before/after → Confirms outcome      │
└───────────────────────┬─────────────────────────────┘
                        │
              ┌─────────┴─────────┐
         success?             failed?
              │                   │
         ADVANCE            HANDLE ERROR
         next step          retry or escalate
```

## Key Design Principles

### 1. Semantic Storage
Never store pixel coordinates. Store element intent.
- Wrong: `{"x": 342, "y": 187}`
- Right: `{"intent": "Click the Submit button", "element_type": "button"}`

### 2. ReAct Agent Loop
Every action follows: Reason → Act → Observe. No blind retries.

### 3. Verify After Every Action
Screenshot diff before/after each action. No assumptions.

### 4. Graceful Degradation
- OmniParser available → pixel-accurate grounding
- OmniParser unavailable → VLM-based coordinate estimation
- Action fails → fallback strategy → human escalation

### 5. Enterprise Security
- Zero data leaves the network (all models local)
- Immutable audit log for every action
- Vault for credential management
- Human-in-the-loop for high-risk actions
