# 🤖 Macro Recorder Agent

> **Automate any interface like a human — using vision, not APIs.**

An intelligent automation agent that records human workflows, replays them, and executes tasks autonomously by **seeing and interacting with the screen**.

---

## ⚡ Why This Project?

Traditional automation is fragile.

It depends on:

* APIs that may not exist
* DOM selectors that break
* App-specific integrations
* Hardcoded coordinates

🚫 This means scripts fail when UI changes.

---

### 💡 Our Approach

Instead of relying on internal app structure, this agent:

* 👁️ **Sees the screen**
* 🧠 **Understands what’s displayed**
* 🎯 **Acts based on intent**
* 🔁 **Adapts to UI changes**

👉 Just like a human would.

---

## 🎬 What It Can Do

### 🎥 Record Workflows

Capture real user behavior:

* Mouse clicks (single / double)
* Keyboard inputs & shortcuts
* Typed text
* Scroll events

Each step includes:

* 📸 Screenshot
* 🧠 Semantic intent

---

### 🔁 Replay Workflows

Two execution modes:

| Mode              | Description                       |
| ----------------- | --------------------------------- |
| **Deterministic** | Replay exact recorded coordinates |
| **Semantic** ⭐    | Re-locate elements using vision   |

👉 Semantic replay makes automation **robust to UI changes**.

---

### 🤖 Run Autonomous Tasks

Give a natural language instruction:

```bash
python main.py --task "Submit my monthly expense report"
```

The agent executes:

```
observe → plan → locate → act → verify
```

---

## 🧠 Core Idea: Intent over Coordinates

Instead of storing:

```json
{"x": 842, "y": 391}
```

We store:

```json
{
  "intent": "Click the Submit button",
  "element_type": "button"
}
```

👉 This makes workflows **portable, resilient, and reusable**.

---

## 🏗️ Architecture

```
User Task
   ↓
Observe Screen
   ↓
Understand UI (Vision Model)
   ↓
Plan Action
   ↓
Locate Element
   ↓
Execute Action
   ↓
Verify Result
```

---

## 📁 Project Structure

```
agent/        → orchestration loop (LangGraph)
recorder/     → event capture & replay engine
vision/       → screen capture + UI grounding
storage/      → memory & audit (PostgreSQL + Qdrant)
execution/    → sandbox / VM layer
monitoring/   → observability (future)
docs/         → architecture notes
data/         → local recordings
main.py       → CLI entry point
```

---

## ⚙️ Tech Stack

| Layer      | Technology            |
| ---------- | --------------------- |
| Language   | Python                |
| Agent      | LangGraph             |
| Vision     | Qwen2.5-VL (via vLLM) |
| UI Parsing | OmniParser            |
| Automation | pyautogui, pynput     |
| Capture    | mss                   |
| Database   | PostgreSQL            |
| Vector DB  | Qdrant                |
| Infra      | Docker Compose        |

---

## 🚀 Quick Start

### 1. Clone the repository

```bash
git clone <your-repo-url>
cd macro-recorder
```

---

### 2. Setup environment

```bash
python -m venv venv
```

Activate:

* Windows:

```bash
venv\Scripts\activate
```

* macOS / Linux:

```bash
source venv/bin/activate
```

---

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

---

### 4. Configure environment

Create `.env` from `.env.example`

```env
VLM_BASE_URL=http://localhost:8000/v1
VLM_MODEL=Qwen2.5-VL
POSTGRES_URL=...
QDRANT_URL=...
```

---

### 5. Start infrastructure

```bash
docker compose up -d
```

Then:

```bash
python scripts/migrate.py
```

---

### 6. Start vision model

```bash
vllm serve Qwen/Qwen2.5-VL-7B-Instruct \
  --host 0.0.0.0 \
  --port 8000
```

---

## ▶️ Usage

### 🤖 Run autonomous task

```bash
python main.py --task "Fill the expense report"
```

---

### 🎥 Record workflow

```bash
python main.py --record --name "expense_report"
```

---

### 🔁 Replay workflow

```bash
python main.py --replay --name "expense_report"
```

---

### 🧭 Smart replay (recommended)

```bash
python main.py --replay --name "expense_report" --semantic
```

---

## 📊 Current Status

### ✅ Working

* Workflow recording
* Screenshot capture
* Replay engine
* Agent loop foundation

### 🚧 In Progress

* Autonomous reliability
* Error recovery strategies
* Observability & monitoring
* Cross-platform support

---

## ⚠️ Limitations

* Currently optimized for **Windows desktop**
* Requires **active GUI session**
* Performance depends on **vision model quality**

---

## 🗺️ Roadmap

* 🧠 Better UI understanding
* 🔁 More robust replay under UI changes
* 🛠️ Advanced action strategies
* 📊 Full observability
* 🧪 End-to-end testing
* 📦 Easy deployment packaging

---

## 🤝 Contributing

Contributions are welcome!

Please:

* Keep PRs focused
* Add tests when possible
* Avoid hardcoded secrets

---

## 📄 License

Coming soon.

---

## ⭐ Vision

> The future of automation is not APIs.
> It's **agents that see, understand, and act like humans.**
