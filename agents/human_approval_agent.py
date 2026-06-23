from langgraph.types import interrupt


def human_approval_node(state: dict) -> dict:
    best_idea = state.get("best_idea", {})

    print(f"\n{'='*60}")
    print("YOUR APPROVAL NEEDED")
    print(f"{'='*60}")
    print(f"\nTitle   : {best_idea.get('title', 'N/A')}")
    print(f"Concept : {best_idea.get('concept', 'N/A')}")
    if best_idea.get("why_best"):
        print(f"Why best: {best_idea.get('why_best', '')}")
    print(f"\n{'='*60}")

    # Pause graph execution — LangGraph surfaces this to the caller in main.py
    decision = interrupt("Do you approve this idea? (yes / no): ")

    approved = str(decision).strip().lower() in ["yes", "y", "approve", "ok"]

    if approved:
        print(f"\n[Approval Agent] Approved! Proceeding to storyboard generation.")
        return {"human_approved": True}
    else:
        print(f"\n[Approval Agent] Rejected. Restarting the full workflow...")
        # Reset all state so research starts fresh
        return {
            "human_approved": False,
            "research_ideas": [],
            "best_idea": None,
            "freshness_approved": None,
            "rejection_reason": "",
            "iteration": 0,
            "image_prompts": [],
        }
