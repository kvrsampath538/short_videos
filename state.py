from typing import TypedDict, Annotated, Optional
from langgraph.graph.message import add_messages


class AgentState(TypedDict):
    # ── Idea generator ──────────────────────────────────────────
    use_idea_generator: bool           # True = run generator, False = use input_topic directly
    generated_ideas: list[dict]        # 3 ideas from idea generator
    selected_idea: Optional[dict]      # idea chosen by user
    idea_gen_iteration: int            # how many batches of ideas generated

    # ── Core pipeline ───────────────────────────────────────────
    input_topic: str
    research_ideas: list[dict]
    freshness_approved: Optional[bool]
    best_idea: Optional[dict]
    rejection_reason: str
    iteration: int

    # ── Output pipeline ─────────────────────────────────────────
    image_prompts: list[dict]
    human_approved: Optional[bool]
    generated_images: list[dict]
    telugu_script: dict

    messages: Annotated[list, add_messages]
