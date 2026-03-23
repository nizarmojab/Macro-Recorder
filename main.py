"""
main.py — Entry point for the Macro Recorder Agent.

Usage:
    # Run agent on a task
    python main.py --task "Fill the monthly expense report"

    # Record a human workflow
    python main.py --record --name "expense_report" --description "Monthly expense submission"

    # Replay a recorded workflow
    python main.py --replay --workflow "expense_report"
"""
import argparse
import sys
from dotenv import load_dotenv

load_dotenv()


def run_agent(task: str):
    from agent.graph import run_task
    print(f"\nRunning agent on task: {task}\n")
    result = run_task(task)
    print(f"\nStatus: {result['status']}")
    print(f"Steps completed: {result['current_step']}")
    if result.get("error"):
        print(f"Error: {result['error']}")


def run_recorder(name: str, description: str):
    from recorder.events.capture import ActionRecorder
    recorder = ActionRecorder(workflow_name=name, task_description=description)
    recorder.start()
    input("Press ENTER to stop recording...\n")
    path = recorder.stop()
    print(f"\nAnnotating recording with VLM (this may take a moment)...")
    recorder.annotate(path)
    print(f"Done. Workflow saved as: {path}")


def run_replay(workflow_name: str, semantic: bool = False):
    from recorder.replay.engine import replay_workflow
    replay_workflow(workflow_name, semantic=semantic)


def main():
    parser = argparse.ArgumentParser(description="Macro Recorder Agent")
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--task", type=str, help="Run agent on a task description")
    mode.add_argument("--record", action="store_true", help="Record a human workflow")
    mode.add_argument("--replay", action="store_true", help="Replay a workflow")

    parser.add_argument("--name", type=str, help="Workflow name (for --record/--replay)")
    parser.add_argument("--description", type=str, default="", help="Task description")
    parser.add_argument("--semantic", action="store_true", help="Use VLM to find elements by label (works even if UI changed)")

    args = parser.parse_args()

    if args.task:
        run_agent(args.task)
    elif args.record:
        if not args.name:
            print("Error: --name is required when recording")
            sys.exit(1)
        run_recorder(args.name, args.description)
    elif args.replay:
        if not args.name:
            print("Error: --name is required when replaying")
            sys.exit(1)
        run_replay(args.name, semantic=args.semantic)


if __name__ == "__main__":
    main()
