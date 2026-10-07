"""Pydantic models for the agent's structured output.

The model doesn't get to hand back free-text recommendations. When it's
ready to recommend something concrete, it calls the propose_recommendation
tool (defined in agent.py) with arguments shaped like Recommendation below.
Pydantic checks the shape (right fields, right types); agent.py then
re-checks the substance (does this activity actually fit) by calling
tools.check_feasibility. Shape and substance are deliberately two separate
checks, because a well-typed recommendation can still be wrong.
"""

from pydantic import BaseModel, Field, field_validator


class Recommendation(BaseModel):
    """A single activity recommendation the agent wants to make.

    This is what propose_recommendation's arguments are validated against
    before agent.py ever calls check_feasibility on them.
    """

    activity_id: str = Field(..., min_length=1)
    day: int = Field(..., ge=1)
    start_time: str = Field(..., min_length=5, max_length=5)
    reason: str = Field(..., min_length=1)

    @field_validator("start_time")
    @classmethod
    def validate_time_format(cls, v: str) -> str:
        if len(v) != 5 or v[2] != ":":
            raise ValueError("start_time must be 'HH:MM', e.g. '14:00'")
        hh, mm = v.split(":")
        if not (hh.isdigit() and mm.isdigit()):
            raise ValueError("start_time must be 'HH:MM', e.g. '14:00'")
        if not (0 <= int(hh) <= 23 and 0 <= int(mm) <= 59):
            raise ValueError("start_time out of range")
        return v


class ValidationOutcome(BaseModel):
    """What agent.py sends back to the model after checking a Recommendation.

    ok=False always carries a reason, so the model has something concrete to
    correct on its retry instead of guessing what went wrong.
    """

    ok: bool
    reason: str | None = None
    details: dict | None = None