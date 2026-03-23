"""
Agent Graph — LangGraph FSM orchestrating the full observe→act→verify loop.
This is the brain of the agent.
"""
from typing import TypedDict, Optional, List
import json
import re

from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver

from vision.capture.screen import capture_screen, capture_before_after
from vision.vlm.client import locate_element, verify_action
from agent.actions.executor import execute_action
from agent.memory.workflow_store import find_similar_workflow, save_workflow_step
from storage.postgres.audit import log_action


def _parse_json(raw: str) -> dict:
    """Parse JSON from VLM response, stripping markdown code blocks if present."""
    text = raw.strip()
    match = re.search(r"```(?:json)?\s*([\s\S]*?)```", text)
    if match:
        text = match.group(1).strip()
    return json.loads(text)


# ─── State Definition ────────────────────────────────────────────────────────

class AgentState(TypedDict):
    task: str                          # High-level task description
    plan: List[dict]                   # Ordered list of planned steps
    current_step: int                  # Index of current step
    screen_state: Optional[dict]       # Last parsed screen state
    last_action: Optional[dict]        # Last action performed
    last_verification: Optional[dict]  # Last verification result
    retry_count: int                   # Retries for current step
    status: str                        # running|completed|failed|waiting_human
    history: List[dict]                # Full execution history
    error: Optional[str]               # Last error message


# ─── Node Functions ───────────────────────────────────────────────────────────

def observe(state: AgentState) -> AgentState:
    """Capture screen — no VLM call, just store the screenshot path."""
    step = state.get("current_step", 0)
    print(f"\n[observe] step {step} — capturing screen...")
    screenshot = capture_screen(prefix="observe")
    return {**state, "screen_state": {"screenshot": str(screenshot)}}


def plan(state: AgentState) -> AgentState:
    """Build execution plan — 1 VLM call combining screen analysis + planning."""
    if state.get("plan"):
        return state  # Plan already exists, skip

    # Check semantic memory first (no VLM needed)
    try:
        similar = find_similar_workflow(state["task"])
    except Exception as e:
        print(f"  [plan] Qdrant unavailable: {e}")
        similar = None

    if similar and similar["similarity"] > 0.85:
        print(f"  [plan] Reusing workflow from memory (similarity={similar['similarity']:.2f})")
        return {**state, "plan": similar["plan"], "current_step": 0}

    # Single VLM call: analyze screen + generate plan together
    from vision.vlm.client import analyze_screen
    screenshot = state["screen_state"].get("screenshot") or str(capture_screen(prefix="plan"))
    prompt = f"""Task: {state['task']}

Look at this screenshot and create a step-by-step plan to complete the task.
Respond ONLY in JSON:
{{
  "steps": [
    {{
      "step": 1,
      "intent": "What we want to achieve",
      "action_type": "click|type|scroll|wait|key",
      "target_description": "Description of the target element",
      "value": "Value to type or key to press (null if not applicable)",
      "fallback": "Alternative if this fails"
    }}
  ]
}}"""
    print(f"  [plan] Generating plan with VLM...")
    raw = analyze_screen(screenshot, prompt, trace_name="plan", max_tokens=2048, json_mode=True)
    try:
        plan_data = _parse_json(raw)
        plan_steps = plan_data.get("steps", [])
        print(f"  [plan] {len(plan_steps)} steps generated")
    except (json.JSONDecodeError, ValueError):
        print(f"  [plan] Parse failed. Raw:\n{raw[:200]}")
        plan_steps = []

    return {**state, "plan": plan_steps, "current_step": 0}


def locate(state: AgentState) -> AgentState:
    """Locate the target element for the current step."""
    if not state["plan"] or state["current_step"] >= len(state["plan"]):
        return {**state, "status": "completed"}

    step = state["plan"][state["current_step"]]
    print(f"[locate] step {state['current_step']}: {step.get('intent', '')} — target: {step.get('target_description', '')}")
    screenshot = capture_screen(prefix="locate")

    raw = locate_element(screenshot, step["target_description"])
    try:
        location_data = _parse_json(raw)
    except (json.JSONDecodeError, ValueError):
        location_data = {"found": False, "error": "parse_failed"}

    updated_step = {**step, "location_data": location_data}
    plan = state["plan"].copy()
    plan[state["current_step"]] = updated_step

    return {**state, "plan": plan}


def act(state: AgentState) -> AgentState:
    """Execute the current step action."""
    if not state["plan"] or state["current_step"] >= len(state["plan"]):
        return {**state, "status": "completed"}
    step = state["plan"][state["current_step"]]

    before, capture_after = capture_before_after(f"step_{state['current_step']}")

    # Log to audit trail BEFORE acting
    log_action(
        task=state["task"],
        step=state["current_step"],
        action=step,
        status="executing",
    )

    print(f"[act] step {state['current_step']}: {step['action_type']} → {step.get('target_description', '')} value={step.get('value', '')}")
    try:
        result = execute_action(
            action_type=step["action_type"],
            target_description=step["target_description"],
            location_data=step.get("location_data"),
            value=step.get("value"),
        )
        after = capture_after()
        success = True
        error = None
        print(f"[act] ✓ success: {result}")
    except Exception as e:
        after = capture_after()
        result = {}
        success = False
        error = str(e)
        print(f"[act] ✗ failed: {e}")

    action_record = {
        "step": state["current_step"],
        "action": step,
        "result": result,
        "success": success,
        "error": error,
        "screenshot_before": str(before),
        "screenshot_after": str(after),
    }

    log_action(
        task=state["task"],
        step=state["current_step"],
        action=step,
        status="completed" if success else "failed",
        error=error,
    )

    return {
        **state,
        "last_action": action_record,
        "history": state["history"] + [action_record],
    }


def verify(state: AgentState) -> AgentState:
    """Verify the last action produced the expected outcome."""
    if state.get("status") == "completed" or not state.get("last_action"):
        return state
    action = state["last_action"]
    if not state["plan"] or state["current_step"] >= len(state["plan"]):
        return {**state, "status": "completed"}
    step = state["plan"][state["current_step"]]

    raw = verify_action(
        before_path=action["screenshot_before"],
        after_path=action["screenshot_after"],
        expected_outcome=step["intent"],
    )

    try:
        verification = _parse_json(raw)
    except (json.JSONDecodeError, ValueError):
        verification = {"success": False, "confidence": 0.0, "parse_error": True}

    print(f"[verify] success={verification.get('success')} confidence={verification.get('confidence', 0):.2f}")
    return {**state, "last_verification": verification}


def handle_error(state: AgentState) -> AgentState:
    """Handle failed steps — retry or escalate."""
    retry_count = state["retry_count"] + 1
    step = state["plan"][state["current_step"]]

    if retry_count >= 3:
        # Escalate to human if fallback also fails
        return {**state, "retry_count": retry_count, "status": "waiting_human"}

    if retry_count == 2 and step.get("fallback"):
        # Try fallback strategy
        plan = state["plan"].copy()
        plan[state["current_step"]] = {
            **step,
            "target_description": step["fallback"],
            "using_fallback": True,
        }
        return {**state, "plan": plan, "retry_count": retry_count}

    return {**state, "retry_count": retry_count}


def advance(state: AgentState) -> AgentState:
    """Move to the next step."""
    next_step = state["current_step"] + 1

    if next_step >= len(state["plan"]):
        save_workflow_step(state["task"], state["plan"], state["history"])
        return {**state, "current_step": next_step, "status": "completed"}

    return {**state, "current_step": next_step, "retry_count": 0}


# ─── Routing Functions ────────────────────────────────────────────────────────

def route_after_verify(state: AgentState) -> str:
    v = state.get("last_verification", {})
    if state["status"] in ("completed", "failed", "waiting_human"):
        return END
    if v.get("success") and v.get("confidence", 0) >= 0.75:
        return "advance"
    return "handle_error"


def route_after_error(state: AgentState) -> str:
    if state["status"] == "waiting_human":
        return END
    return "locate"


# ─── Graph Assembly ───────────────────────────────────────────────────────────

def build_agent_graph():
    graph = StateGraph(AgentState)

    graph.add_node("observe", observe)
    graph.add_node("plan", plan)
    graph.add_node("locate", locate)
    graph.add_node("act", act)
    graph.add_node("verify", verify)
    graph.add_node("handle_error", handle_error)
    graph.add_node("advance", advance)

    graph.set_entry_point("observe")
    graph.add_edge("observe", "plan")
    graph.add_edge("plan", "locate")
    graph.add_edge("locate", "act")
    graph.add_edge("act", "verify")
    graph.add_conditional_edges("verify", route_after_verify, {
        "advance": "advance",
        "handle_error": "handle_error",
        END: END,
    })
    graph.add_conditional_edges("handle_error", route_after_error, {
        "locate": "locate",
        END: END,
    })
    graph.add_edge("advance", "observe")

    checkpointer = MemorySaver()
    return graph.compile(checkpointer=checkpointer)


agent = build_agent_graph()


def run_task(task: str, thread_id: str = "default") -> dict:
    """Run the agent on a task."""
    initial_state: AgentState = {
        "task": task,
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

    config = {"configurable": {"thread_id": thread_id}}
    final_state = agent.invoke(initial_state, config=config)
    return final_state
