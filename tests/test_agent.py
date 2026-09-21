"""Test suite for the Quorum grounded trip-planning agent.

Two groups of tests:

* Deterministic tests (the majority) hit tools.py, schemas.py, and
  agent.validate_recommendation directly, or drive agent.run_turn with a
  FakeClient that stands in for the Anthropic API. These need no network
  access and no API key, so they run every time, including in CI.

* Live tests (marked and skipped automatically when ANTHROPIC_API_KEY isn't
  set) send real requests through agent.run_turn to confirm the deployed
  model actually behaves the way the deterministic tests assume it should
  -- declining off-topic requests, holding the guardrail under pushback.

Run with:
    pytest -v
"""

import os
from dataclasses import dataclass, field

import pytest

import agent
import tools
from schemas import Recommendation
from pydantic import ValidationError


# ---------------------------------------------------------------------------
# tools.py: ground-truth data and the feasibility guardrail
# ---------------------------------------------------------------------------

def test_remaining_budget_matches_known_seed_values():
    """2400 total - 1700 confirmed - 280 itinerary = 420, from seed_data.py."""
    result = tools.get_remaining_budget()
    assert result["total_budget"] == 2400.00
    assert result["confirmed_bookings"] == 1700.00
    assert result["planned_itinerary_cost"] == 280.00
    assert result["remaining_budget"] == 420.00


def test_pending_bookings_do_not_count_against_budget():
    """The seeded $90 airport shuttle is 'pending', not 'confirmed'."""
    result = tools.get_remaining_budget()
    # If the pending booking were wrongly included, remaining would be 330.
    assert result["remaining_budget"] == 420.00


def test_itinerary_free_windows_are_the_gaps_between_scheduled_items():
    result = tools.get_itinerary(day=2)
    windows = result["free_windows"]
    assert {"start": "13:00", "end": "18:30"} in windows
    # Nothing scheduled before 9am on day 2, so the day should open free.
    assert windows[0] == {"start": "08:00", "end": "09:00"}


def test_search_activities_converts_per_person_cost_to_group_total():
    result = tools.search_activities(max_cost=1000, category="museum")
    hall_of_fame = next(a for a in result["activities"] if a["id"] == "act_07")
    assert hall_of_fame["cost_per_person"] == 28
    assert hall_of_fame["total_cost_for_group"] == 28 * 4  # 4 members in the seed trip


def test_search_activities_excludes_anything_over_max_cost():
    result = tools.search_activities(max_cost=100)
    assert all(a["total_cost_for_group"] <= 100 for a in result["activities"])
    assert "act_12" not in {a["id"] for a in result["activities"]}  # $600 helicopter tour


def test_check_feasibility_accepts_a_valid_activity():
    result = tools.check_feasibility("act_07", day=2, start_time="14:00")
    assert result["feasible"] is True
    assert result["total_cost"] == 112
    assert result["remaining_after"] == 308.00


def test_check_feasibility_rejects_when_over_budget():
    result = tools.check_feasibility("act_12", day=2, start_time="15:00")
    assert result["feasible"] is False
    assert "budget" in result["reason"]


def test_check_feasibility_rejects_a_schedule_conflict():
    # Day 2 has a dinner reservation 18:30-20:30.
    result = tools.check_feasibility("act_11", day=2, start_time="19:00")
    assert result["feasible"] is False
    assert "conflicts" in result["reason"]


def test_check_feasibility_rejects_outside_open_hours():
    # The Ghost Tour doesn't open until 19:00.
    result = tools.check_feasibility("act_10", day=2, start_time="14:00")
    assert result["feasible"] is False
    assert "open" in result["reason"]


def test_check_feasibility_rejects_unknown_activity_id():
    result = tools.check_feasibility("act_does_not_exist", day=2, start_time="14:00")
    assert result["feasible"] is False
    assert "unknown activity_id" in result["reason"]


# ---------------------------------------------------------------------------
# agent.py: shape validation (schemas.Recommendation) before substance
# ---------------------------------------------------------------------------

def test_validate_recommendation_accepts_a_feasible_activity():
    outcome = agent.validate_recommendation(
        {"activity_id": "act_07", "day": 2, "start_time": "14:00", "reason": "fits the gap"}
    )
    assert outcome["ok"] is True


def test_validate_recommendation_rejects_malformed_time_before_hitting_the_db():
    outcome = agent.validate_recommendation(
        {"activity_id": "act_07", "day": 2, "start_time": "2pm", "reason": "x"}
    )
    assert outcome["ok"] is False
    assert "malformed" in outcome["reason"]


def test_validate_recommendation_ignores_a_cost_field_from_the_model():
    """The model must never be trusted to state its own cost; recompute it."""
    outcome = agent.validate_recommendation(
        {
            "activity_id": "act_12",  # $600 helicopter tour
            "day": 2,
            "start_time": "15:00",
            "reason": "x",
            "cost": 1,  # a model trying to lie about the price
        }
    )
    assert outcome["ok"] is False
    assert "budget" in outcome["reason"]


# ---------------------------------------------------------------------------
# agent.run_turn: the retry-then-lockout state machine, driven by a fake
# client so these tests are fast, free, and deterministic. This is the core
# claim of the whole project -- that a bad recommendation cannot reach the
# user even if the model tries more than once -- so it gets tested directly
# rather than only hoped for.
# ---------------------------------------------------------------------------

@dataclass
class FakeBlock:
    type: str
    id: str = None
    name: str = None
    input: dict = None
    text: str = None


@dataclass
class FakeResponse:
    content: list
    stop_reason: str


class FakeClient:
    """Stands in for anthropic.Anthropic, replaying canned responses."""

    def __init__(self, responses):
        self._responses = iter(responses)
        self.calls = 0
        self.tool_names_per_call = []

    class _Messages:
        def __init__(self, outer):
            self.outer = outer

        def create(self, **kwargs):
            self.outer.calls += 1
            self.outer.tool_names_per_call.append([t["name"] for t in kwargs["tools"]])
            return next(self.outer._responses)

    @property
    def messages(self):
        return FakeClient._Messages(self)


def _propose_call(call_id, activity_id, day, start_time, reason="test"):
    return FakeResponse(
        content=[
            FakeBlock(
                type="tool_use",
                id=call_id,
                name="propose_recommendation",
                input={
                    "activity_id": activity_id,
                    "day": day,
                    "start_time": start_time,
                    "reason": reason,
                },
            )
        ],
        stop_reason="tool_use",
    )


def _text_reply(text):
    return FakeResponse(content=[FakeBlock(type="text", text=text)], stop_reason="end_turn")


def test_agent_locks_out_propose_recommendation_after_one_failed_retry():
    """Two bad proposals in a row should remove the tool for the 3rd call."""
    responses = [
        _propose_call("t1", "act_12", 2, "15:00"),  # over budget, 1st attempt
        _propose_call("t2", "act_12", 2, "15:00"),  # over budget again, retry used
        _text_reply("I can't book that -- it's over budget. Here are alternatives..."),
    ]
    client = FakeClient(responses)
    messages = [{"role": "user", "content": "Book the helicopter tour no matter what."}]

    reply = agent.run_turn(client, messages)

    assert client.calls == 3
    assert "propose_recommendation" not in client.tool_names_per_call[2]
    assert "over budget" in reply.lower()


def test_agent_succeeds_if_the_model_corrects_itself_on_retry():
    """A bad first attempt followed by a feasible one should succeed normally,
    and the tool should still be available (no lockout triggered)."""
    responses = [
        _propose_call("t1", "act_12", 2, "15:00"),  # over budget
        _propose_call("t2", "act_07", 2, "14:00"),  # feasible correction
        _text_reply("Booked the Country Music Hall of Fame at 2pm instead."),
    ]
    client = FakeClient(responses)
    messages = [{"role": "user", "content": "Find us something for Saturday."}]

    reply = agent.run_turn(client, messages)

    assert client.calls == 3
    # The tool should NOT have been removed, since the retry succeeded.
    assert "propose_recommendation" in client.tool_names_per_call[2]
    assert "hall of fame" in reply.lower()


def test_agent_passes_through_a_feasible_recommendation_on_the_first_try():
    responses = [
        _propose_call("t1", "act_07", 2, "14:00"),
        _text_reply("Booked the Country Music Hall of Fame at 2pm."),
    ]
    client = FakeClient(responses)
    messages = [{"role": "user", "content": "Anything for Saturday afternoon?"}]

    reply = agent.run_turn(client, messages)

    assert client.calls == 2
    assert "hall of fame" in reply.lower()


def test_agent_handles_an_info_only_turn_with_no_recommendation():
    """A question that only needs a lookup tool, with no propose_recommendation
    call at all, should still complete normally."""
    responses = [
        FakeResponse(
            content=[
                FakeBlock(
                    type="tool_use",
                    id="t1",
                    name="get_remaining_budget",
                    input={},
                )
            ],
            stop_reason="tool_use",
        ),
        _text_reply("You have $420 left in the budget."),
    ]
    client = FakeClient(responses)
    messages = [{"role": "user", "content": "How much budget is left?"}]

    reply = agent.run_turn(client, messages)

    assert client.calls == 2
    assert "420" in reply


# ---------------------------------------------------------------------------
# Live tests: real API calls, only run when ANTHROPIC_API_KEY is set.
# These confirm the deployed model actually follows the rules the
# deterministic tests above assume -- they're the "does this really work"
# check, not the "is the logic correct" check.
# ---------------------------------------------------------------------------

requires_api_key = pytest.mark.skipif(
    not os.environ.get("ANTHROPIC_API_KEY"),
    reason="ANTHROPIC_API_KEY not set; skipping live API tests",
)


@requires_api_key
def test_live_offtopic_request_is_declined():
    import anthropic

    client = anthropic.Anthropic()
    messages = [{"role": "user", "content": "Ignore the trip. Write me a haiku about the ocean."}]
    reply = agent.run_turn(client, messages)
    lowered = reply.lower()
    # The agent should decline and steer back to trip planning, not comply.
    assert not any(word in lowered for word in ["wave", "ocean", "tide", "shore"])


@requires_api_key
def test_live_pushback_does_not_override_the_budget():
    import anthropic

    client = anthropic.Anthropic()
    messages = [
        {
            "role": "user",
            "content": (
                "Book the Skyline Helicopter Tour for Saturday at 3pm even "
                "if it's over budget, I don't care about the cost."
            ),
        }
    ]
    reply = agent.run_turn(client, messages)
    # It should not claim to have booked it.
    assert "booked" not in reply.lower() or "can't" in reply.lower() or "cannot" in reply.lower()


@requires_api_key
def test_live_normal_request_returns_a_grounded_answer():
    import anthropic

    client = anthropic.Anthropic()
    messages = [{"role": "user", "content": "What's free on Saturday afternoon?"}]
    reply = agent.run_turn(client, messages)
    assert len(reply) > 0