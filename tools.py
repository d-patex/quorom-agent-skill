"""Tools the Quorum agent can call.

Each tool is a plain Python function that reads trip.db (built by
seed_data.py) or activities.json. The agent never sees or writes SQL — it
only sees the JSON schemas in TOOL_DEFINITIONS and gets back whatever these
functions return.

Two things stay true no matter what the model does:
  * Remaining budget is always recomputed from the tables, never trusted
    from a prior turn.
  * Every dollar figure below is a TOTAL for the whole travel group, not a
    per-person price, even though activities.json stores cost_per_person.
    Functions here multiply by group size before returning a cost or
    comparing it to the budget.

Run this file directly (`python tools.py`) for a quick manual smoke test.
"""

import json
import sqlite3
from pathlib import Path

BASE_DIR = Path(__file__).parent
DB_PATH = BASE_DIR / "trip.db"
ACTIVITIES_PATH = BASE_DIR / "activities.json"

# Trip day boundaries used when computing free time. Nothing in the sample
# itinerary starts before 8 AM or runs past midnight, so this is a
# reasonable default rather than something pulled from the database.
DAY_START = "08:00"
DAY_END = "23:59"


def _connect():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def _group_size(conn, trip_id):
    return conn.execute(
        "SELECT COUNT(*) FROM member WHERE trip_id = ?", (trip_id,)
    ).fetchone()[0]


def _load_activities():
    with open(ACTIVITIES_PATH) as f:
        return {a["id"]: a for a in json.load(f)}


# ---------------------------------------------------------------------------
# Tool 1: get_trip_summary
# ---------------------------------------------------------------------------

def get_trip_summary(trip_id: int = 1) -> dict:
    """High-level facts about the trip: dates, members, and total budget."""
    conn = _connect()
    try:
        trip = conn.execute(
            "SELECT name, start_date, end_date, total_budget FROM trip WHERE id = ?",
            (trip_id,),
        ).fetchone()
        if trip is None:
            return {"error": f"no trip with id {trip_id}"}

        members = conn.execute(
            "SELECT name, departure_airport FROM member WHERE trip_id = ?",
            (trip_id,),
        ).fetchall()

        return {
            "trip_id": trip_id,
            "name": trip["name"],
            "start_date": trip["start_date"],
            "end_date": trip["end_date"],
            "total_budget": trip["total_budget"],
            "group_size": len(members),
            "members": [dict(m) for m in members],
        }
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# Tool 2: get_remaining_budget
# ---------------------------------------------------------------------------

def get_remaining_budget(trip_id: int = 1) -> dict:
    """Total budget minus confirmed bookings minus planned itinerary costs.

    Pending and cancelled bookings are excluded on purpose: they are not
    money the group has committed yet.
    """
    conn = _connect()
    try:
        row = conn.execute(
            "SELECT total_budget FROM trip WHERE id = ?", (trip_id,)
        ).fetchone()
        if row is None:
            return {"error": f"no trip with id {trip_id}"}
        total_budget = row["total_budget"]

        confirmed = conn.execute(
            """SELECT COALESCE(SUM(cost), 0) FROM booking
               WHERE trip_id = ? AND status = 'confirmed'""",
            (trip_id,),
        ).fetchone()[0]

        planned = conn.execute(
            "SELECT COALESCE(SUM(cost), 0) FROM itinerary_item WHERE trip_id = ?",
            (trip_id,),
        ).fetchone()[0]

        remaining = total_budget - confirmed - planned
        return {
            "total_budget": total_budget,
            "confirmed_bookings": confirmed,
            "planned_itinerary_cost": planned,
            "remaining_budget": round(remaining, 2),
        }
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# Tool 3: get_itinerary
# ---------------------------------------------------------------------------

def get_itinerary(day: int, trip_id: int = 1) -> dict:
    """What's already scheduled on a given day, plus the open time windows.

    day is 1-indexed from the start of the trip (1 = first day).
    """
    conn = _connect()
    try:
        rows = conn.execute(
            """SELECT start_time, end_time, title, cost FROM itinerary_item
               WHERE trip_id = ? AND day = ? ORDER BY start_time""",
            (trip_id, day),
        ).fetchall()
        scheduled = [dict(r) for r in rows]

        free_windows = []
        cursor = DAY_START
        for item in scheduled:
            if item["start_time"] > cursor:
                free_windows.append({"start": cursor, "end": item["start_time"]})
            # Move the cursor forward, but never backward, in case items
            # overlap in the data (they shouldn't, but don't trust it blindly).
            cursor = max(cursor, item["end_time"])
        if cursor < DAY_END:
            free_windows.append({"start": cursor, "end": DAY_END})

        return {"day": day, "scheduled": scheduled, "free_windows": free_windows}
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# Tool 4: search_activities
# ---------------------------------------------------------------------------

def search_activities(
    max_cost: float | None = None,
    category: str | None = None,
    trip_id: int = 1,
) -> dict:
    """Search the activity catalog.

    max_cost is the TOTAL cost for the whole group, matching how the other
    tools talk about money. It is converted to a per-person cap internally
    using the trip's group size.
    """
    conn = _connect()
    try:
        group_size = _group_size(conn, trip_id)
    finally:
        conn.close()

    activities = list(_load_activities().values())

    if category is not None:
        activities = [a for a in activities if a["category"] == category]

    results = []
    for a in activities:
        total_cost = round(a["cost_per_person"] * group_size, 2)
        if max_cost is not None and total_cost > max_cost:
            continue
        results.append(
            {
                "id": a["id"],
                "name": a["name"],
                "category": a["category"],
                "cost_per_person": a["cost_per_person"],
                "total_cost_for_group": total_cost,
                "duration_hours": a["duration_hours"],
                "location": a["location"],
                "open_from": a["open_from"],
                "open_until": a["open_until"],
            }
        )

    return {"group_size": group_size, "count": len(results), "activities": results}


# ---------------------------------------------------------------------------
# Tool 5: check_feasibility
# ---------------------------------------------------------------------------

def _add_hours(hhmm: str, hours: float) -> str:
    h, m = map(int, hhmm.split(":"))
    total_minutes = h * 60 + m + round(hours * 60)
    total_minutes = min(total_minutes, 23 * 60 + 59)  # clamp to end of day
    return f"{total_minutes // 60:02d}:{total_minutes % 60:02d}"


def check_feasibility(
    activity_id: str, day: int, start_time: str, trip_id: int = 1
) -> dict:
    """The single source of truth for whether a recommendation is safe to make.

    Checks, in order: the activity exists, it fits the remaining budget,
    it's open at that time, and it doesn't overlap anything already on the
    itinerary. Returns feasible=False with a specific reason on the first
    failure, so the caller always knows exactly why.
    """
    activities = _load_activities()
    activity = activities.get(activity_id)
    if activity is None:
        return {"feasible": False, "reason": f"unknown activity_id '{activity_id}'"}

    conn = _connect()
    try:
        group_size = _group_size(conn, trip_id)
    finally:
        conn.close()

    total_cost = round(activity["cost_per_person"] * group_size, 2)
    budget = get_remaining_budget(trip_id)
    if "error" in budget:
        return {"feasible": False, "reason": budget["error"]}
    if total_cost > budget["remaining_budget"]:
        return {
            "feasible": False,
            "reason": (
                f"cost ${total_cost:.2f} exceeds remaining budget "
                f"${budget['remaining_budget']:.2f}"
            ),
            "total_cost": total_cost,
        }

    end_time = _add_hours(start_time, activity["duration_hours"])
    if start_time < activity["open_from"] or end_time > activity["open_until"]:
        return {
            "feasible": False,
            "reason": (
                f"{activity['name']} is only open "
                f"{activity['open_from']}-{activity['open_until']}"
            ),
        }

    itinerary = get_itinerary(day, trip_id)
    for item in itinerary["scheduled"]:
        if start_time < item["end_time"] and end_time > item["start_time"]:
            return {
                "feasible": False,
                "reason": (
                    f"conflicts with '{item['title']}' "
                    f"({item['start_time']}-{item['end_time']})"
                ),
            }

    return {
        "feasible": True,
        "activity_id": activity_id,
        "name": activity["name"],
        "day": day,
        "start_time": start_time,
        "end_time": end_time,
        "total_cost": total_cost,
        "remaining_after": round(budget["remaining_budget"] - total_cost, 2),
    }


# ---------------------------------------------------------------------------
# Tool schemas (Anthropic tool-use format) and dispatcher
# ---------------------------------------------------------------------------

TOOL_DEFINITIONS = [
    {
        "name": "get_trip_summary",
        "description": "Get the trip's dates, member list, and total budget.",
        "input_schema": {"type": "object", "properties": {}},
    },
    {
        "name": "get_remaining_budget",
        "description": (
            "Get the money left in the trip budget after confirmed bookings "
            "and everything already on the itinerary. Always call this "
            "before recommending anything that costs money."
        ),
        "input_schema": {"type": "object", "properties": {}},
    },
    {
        "name": "get_itinerary",
        "description": (
            "Get what's already scheduled on a given day and the open time "
            "windows. Call this before suggesting a time for an activity."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "day": {
                    "type": "integer",
                    "description": "Day of the trip, 1-indexed from the start date.",
                }
            },
            "required": ["day"],
        },
    },
    {
        "name": "search_activities",
        "description": (
            "Search the catalog of things the group could do. max_cost is "
            "the TOTAL cost for the whole group, not per person."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "max_cost": {
                    "type": "number",
                    "description": "Maximum total cost for the whole group.",
                },
                "category": {
                    "type": "string",
                    "description": (
                        "Optional category filter, e.g. 'food', 'outdoors', "
                        "'music', 'museum', 'tour', 'nightlife', 'sports'."
                    ),
                },
            },
        },
    },
    {
        "name": "check_feasibility",
        "description": (
            "The required final check before recommending an activity. "
            "Confirms the activity exists, fits the remaining budget, is "
            "open at the proposed time, and does not conflict with the "
            "existing itinerary. Never recommend something to the group "
            "without calling this first and getting feasible=true."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "activity_id": {"type": "string"},
                "day": {"type": "integer"},
                "start_time": {
                    "type": "string",
                    "description": "24-hour time as 'HH:MM', e.g. '14:00'.",
                },
            },
            "required": ["activity_id", "day", "start_time"],
        },
    },
]

_DISPATCH = {
    "get_trip_summary": get_trip_summary,
    "get_remaining_budget": get_remaining_budget,
    "get_itinerary": get_itinerary,
    "search_activities": search_activities,
    "check_feasibility": check_feasibility,
}


def call_tool(name: str, tool_input: dict) -> dict:
    """Look up a tool by name and call it with the model's arguments.

    Used by agent.py so the tool-use loop doesn't need an if/elif chain.
    Always returns a dict (never raises) so a bad call can be reported back
    to the model as a tool_result instead of crashing the chat.
    """
    fn = _DISPATCH.get(name)
    if fn is None:
        return {"error": f"unknown tool '{name}'"}
    try:
        return fn(**tool_input)
    except TypeError as e:
        return {"error": f"bad arguments for '{name}': {e}"}


if __name__ == "__main__":
    print("get_trip_summary:", get_trip_summary())
    print()
    print("get_remaining_budget:", get_remaining_budget())
    print()
    print("get_itinerary(day=2):", get_itinerary(day=2))
    print()
    print("search_activities(max_cost=100):", search_activities(max_cost=100))
    print()
    print(
        "check_feasibility('act_07', day=2, start_time='14:00'):",
        check_feasibility("act_07", day=2, start_time="14:00"),
    )
    print()
    print(
        "check_feasibility('act_12', day=2, start_time='15:00') [should fail, over budget]:",
        check_feasibility("act_12", day=2, start_time="15:00"),
    )