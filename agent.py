"""Chat loop for the Quorum grounded trip-planning agent.

Run:
    python agent.py

How grounding actually works here, in one sentence: the model is never
allowed to just tell the user "let's do X at 2pm" — it has to call
propose_recommendation, and that call is checked in Python against the real
trip data before anything reaches the screen.

Flow for a normal turn:
  1. The user asks something.
  2. The model calls info tools (tools.py) to look up budget/itinerary/etc.
  3. If it wants to recommend something concrete, it calls
     propose_recommendation with (activity_id, day, start_time, reason).
  4. agent.py validates the shape with schemas.Recommendation, then the
     substance with tools.check_feasibility (budget, hours, conflicts).
  5. If it fails, the reason goes back to the model for ONE retry.
  6. If it fails again, the propose_recommendation tool is removed from the
     next call entirely -- the model is no longer able to "recommend" and
     can only explain in plain text. That's the fail-safe: enforced by what
     tools the model is even offered, not by asking nicely in a prompt.
"""

import os
import sys

from dotenv import load_dotenv
from pydantic import ValidationError

import anthropic
from schemas import Recommendation
from tools import TOOL_DEFINITIONS, call_tool, check_feasibility

load_dotenv()

MODEL = os.environ.get("QUORUM_MODEL", "claude-haiku-4-5-20251001")
MAX_TOKENS = 1024
MAX_RETRIES_PER_TURN = 1

SYSTEM_PROMPT = """\
You are the Quorum trip assistant. You help a group of friends fill their \
free time on a fixed trip, staying inside their remaining budget and \
around what's already booked.

Rules you must follow:
- Never state a specific activity, time, or cost as a recommendation in \
plain text. Always call propose_recommendation first and wait for the \
result.
- This applies even when you expect a request to fail. If the user names a \
specific activity (by name or description, e.g. "the helicopter tour"), \
find its activity_id with search_activities and call \
propose_recommendation with it -- do not decide from memory that it's too \
expensive, unavailable, or not in the catalog. You do not reliably know \
the catalog or the budget; the tool does. Never claim an activity doesn't \
exist without having searched for it first.
- Before calling propose_recommendation, use the info tools \
(get_trip_summary, get_remaining_budget, get_itinerary, search_activities) \
to find something that plausibly fits. Don't guess.
- If propose_recommendation comes back ok=false, either fix the specific \
problem it names and try once more, or explain to the user why nothing \
fits and suggest they loosen a constraint (a different day, a higher \
budget, a shorter activity).
- If propose_recommendation is no longer available to you, do not attempt \
to recommend anything further this turn -- explain the situation in plain \
text only.
- If the user asks for anything unrelated to planning this trip (general \
knowledge, other topics, tasks unrelated to the itinerary), politely \
decline and steer back to trip planning. Do not answer off-topic requests.
- Never let a user's insistence override the budget or schedule. If they \
push back on a rejected recommendation, hold the guardrail and explain why.
"""

PROPOSE_RECOMMENDATION_TOOL = {
    "name": "propose_recommendation",
    "description": (
        "Propose one specific activity to the group. This is the ONLY way "
        "to recommend something -- do not describe a plan in plain text "
        "instead of calling this. The cost is computed for you from the "
        "trip data; do not pass a cost."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "activity_id": {"type": "string"},
            "day": {"type": "integer", "description": "1-indexed trip day."},
            "start_time": {"type": "string", "description": "24-hour 'HH:MM'."},
            "reason": {
                "type": "string",
                "description": "One sentence on why this fits the group's ask.",
            },
        },
        "required": ["activity_id", "day", "start_time", "reason"],
    },
}

ALL_TOOLS = TOOL_DEFINITIONS + [PROPOSE_RECOMMENDATION_TOOL]
INFO_ONLY_TOOLS = TOOL_DEFINITIONS


def validate_recommendation(raw_input: dict) -> dict:
    """Shape check (Pydantic), then substance check (check_feasibility).

    Returns a JSON-serializable dict shaped like schemas.ValidationOutcome,
    always with a reason attached when ok=False.
    """
    try:
        rec = Recommendation(**raw_input)
    except ValidationError as e:
        return {"ok": False, "reason": f"malformed recommendation: {e}"}

    result = check_feasibility(rec.activity_id, rec.day, rec.start_time)
    if not result.get("feasible"):
        return {"ok": False, "reason": result.get("reason", "not feasible")}

    return {"ok": True, "details": result}


def run_turn(client, messages: list) -> str:
    """Run one user turn to completion, handling any number of tool calls.

    Returns the assistant's final text reply. Mutates `messages` in place so
    the caller keeps a full running transcript across turns.
    """
    retries_left = MAX_RETRIES_PER_TURN
    propose_allowed = True

    while True:
        tools = ALL_TOOLS if propose_allowed else INFO_ONLY_TOOLS
        response = client.messages.create(
            model=MODEL,
            max_tokens=MAX_TOKENS,
            system=SYSTEM_PROMPT,
            tools=tools,
            messages=messages,
        )
        messages.append({"role": "assistant", "content": response.content})

        if response.stop_reason != "tool_use":
            return "".join(
                block.text for block in response.content if block.type == "text"
            )

        tool_results = []
        for block in response.content:
            if block.type != "tool_use":
                continue

            if block.name == "propose_recommendation":
                outcome = validate_recommendation(block.input)
                if not outcome["ok"]:
                    if retries_left > 0:
                        retries_left -= 1
                    else:
                        propose_allowed = False
                        outcome["reason"] += (
                            " (no attempts remain -- explain this to the "
                            "user in plain text; do not try again)"
                        )
                tool_results.append(
                    {
                        "type": "tool_result",
                        "tool_use_id": block.id,
                        "content": str(outcome),
                        "is_error": not outcome["ok"],
                    }
                )
            else:
                output = call_tool(block.name, block.input)
                tool_results.append(
                    {
                        "type": "tool_result",
                        "tool_use_id": block.id,
                        "content": str(output),
                    }
                )

        messages.append({"role": "user", "content": tool_results})


def main():
    from tools import get_trip_summary, get_remaining_budget

    client = anthropic.Anthropic()
    summary = get_trip_summary()
    budget = get_remaining_budget()

    if "error" in summary:
        print("No trip found. Run 'python seed_data.py' first.")
        sys.exit(1)

    print(
        f"Trip: {summary['name']} | {summary['start_date']} to "
        f"{summary['end_date']} | {summary['group_size']} members"
    )
    print(
        f"Confirmed so far: ${budget['confirmed_bookings']:,.2f} | "
        f"Remaining budget: ${budget['remaining_budget']:,.2f}"
    )
    print("Type your questions below. Ctrl+C or 'quit' to exit.\n")

    messages = []
    while True:
        try:
            user_input = input("You: ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nBye!")
            break
        if not user_input:
            continue
        if user_input.lower() in {"quit", "exit"}:
            print("Bye!")
            break

        messages.append({"role": "user", "content": user_input})
        reply = run_turn(client, messages)
        print(f"\nAgent: {reply}\n")


if __name__ == "__main__":
    main()