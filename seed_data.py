"""Build the sample trip database that the Quorum agent is grounded in.

Run from the repo root:
    python seed_data.py

The database is dropped and recreated on every run, so you always start from
the same known state. That makes tests and demos repeatable.

Design notes
------------
* Remaining budget is NEVER stored. It is computed from the tables, so it can't
  drift out of sync with the bookings and itinerary it is derived from:
      remaining = total_budget
                  - SUM(confirmed bookings)
                  - SUM(itinerary item costs)
  Pending and cancelled bookings do not count against the budget.
* All times are 24-hour zero-padded "HH:MM" strings, so plain string comparison
  ("09:00" < "14:30") is also correct time comparison.
* Itinerary costs are TOTAL cost for the whole group. Catalog activities in
  activities.json list cost_per_person, so multiply by group size before
  comparing to the budget.
"""

import json
import sqlite3
from pathlib import Path

BASE_DIR = Path(__file__).parent
DB_PATH = BASE_DIR / "trip.db"
ACTIVITIES_PATH = BASE_DIR / "activities.json"

SCHEMA = """
PRAGMA foreign_keys = ON;

DROP TABLE IF EXISTS itinerary_item;
DROP TABLE IF EXISTS booking;
DROP TABLE IF EXISTS member;
DROP TABLE IF EXISTS trip;

CREATE TABLE trip (
    id            INTEGER PRIMARY KEY,
    name          TEXT NOT NULL,
    start_date    TEXT NOT NULL,                    -- ISO 8601, e.g. 2026-10-09
    end_date      TEXT NOT NULL,
    total_budget  REAL NOT NULL CHECK (total_budget >= 0)
);

CREATE TABLE member (
    id                 INTEGER PRIMARY KEY,
    trip_id            INTEGER NOT NULL REFERENCES trip(id),
    name               TEXT NOT NULL,
    departure_airport  TEXT NOT NULL
);

CREATE TABLE booking (
    id           INTEGER PRIMARY KEY,
    trip_id      INTEGER NOT NULL REFERENCES trip(id),
    type         TEXT NOT NULL CHECK (type IN ('flight', 'lodging', 'transport', 'other')),
    description  TEXT NOT NULL,
    cost         REAL NOT NULL CHECK (cost >= 0),
    status       TEXT NOT NULL CHECK (status IN ('confirmed', 'pending', 'cancelled'))
);

CREATE TABLE itinerary_item (
    id          INTEGER PRIMARY KEY,
    trip_id     INTEGER NOT NULL REFERENCES trip(id),
    day         INTEGER NOT NULL CHECK (day >= 1),  -- 1 = first day of the trip
    start_time  TEXT NOT NULL,                      -- "HH:MM", 24-hour
    end_time    TEXT NOT NULL,
    title       TEXT NOT NULL,
    cost        REAL NOT NULL DEFAULT 0 CHECK (cost >= 0),
    CHECK (start_time < end_time)
);
"""

# ---------------------------------------------------------------------------
# Sample data: a 3-day weekend in Nashville for 4 friends.
# Oct 9, 2026 is a Friday, so day 1 = Fri, day 2 = Sat, day 3 = Sun.
# ---------------------------------------------------------------------------

TRIP = (1, "Nashville Weekend", "2026-10-09", "2026-10-11", 2400.00)

MEMBERS = [
    (1, 1, "Alex", "IND"),
    (2, 1, "Priya", "ORD"),
    (3, 1, "Sam", "CMH"),
    (4, 1, "Jordan", "DTW"),
]

BOOKINGS = [
    # (id, trip_id, type, description, cost, status)
    (1, 1, "flight", "Round-trip flights, 4 travelers", 780.00, "confirmed"),
    (2, 1, "lodging", "Rental house, 2 nights", 800.00, "confirmed"),
    (3, 1, "transport", "Rental van, 3 days", 120.00, "confirmed"),
    # Pending on purpose: it must NOT count against the budget yet.
    (4, 1, "transport", "Airport shuttle (quote only)", 90.00, "pending"),
]

ITINERARY = [
    # (id, trip_id, day, start, end, title, cost)
    # Day 1 (Fri)
    (1, 1, 1, "16:00", "17:00", "Arrive and check in", 0.00),
    (2, 1, 1, "19:00", "21:00", "Group dinner", 120.00),
    # Day 2 (Sat)
    (3, 1, 2, "09:00", "10:00", "Breakfast at the rental house", 0.00),
    (4, 1, 2, "10:30", "13:00", "Free walking tour of downtown", 0.00),
    (5, 1, 2, "18:30", "20:30", "Dinner reservation", 160.00),
    # Day 3 (Sun)
    (6, 1, 3, "10:00", "11:00", "Check out and pack", 0.00),
    (7, 1, 3, "14:30", "15:30", "Airport drop-off", 0.00),
]

REQUIRED_ACTIVITY_KEYS = {
    "id": str,
    "name": str,
    "category": str,
    "cost_per_person": (int, float),
    "duration_hours": (int, float),
    "location": str,
    "open_from": str,
    "open_until": str,
}


def create_database(db_path=DB_PATH):
    """Drop, recreate, and populate the trip database at db_path."""
    conn = sqlite3.connect(db_path)
    try:
        conn.executescript(SCHEMA)
        conn.execute("INSERT INTO trip VALUES (?, ?, ?, ?, ?)", TRIP)
        conn.executemany("INSERT INTO member VALUES (?, ?, ?, ?)", MEMBERS)
        conn.executemany("INSERT INTO booking VALUES (?, ?, ?, ?, ?, ?)", BOOKINGS)
        conn.executemany(
            "INSERT INTO itinerary_item VALUES (?, ?, ?, ?, ?, ?, ?)", ITINERARY
        )
        conn.commit()
    finally:
        conn.close()


def validate_activities(path=ACTIVITIES_PATH):
    """Load activities.json and check every entry is well-formed.

    Raises ValueError with a specific message on the first problem, so a typo
    in the catalog fails loudly here instead of confusing the agent later.
    """
    with open(path) as f:
        activities = json.load(f)

    seen_ids = set()
    for act in activities:
        for key, expected_type in REQUIRED_ACTIVITY_KEYS.items():
            if key not in act:
                raise ValueError(f"{act.get('id', '?')}: missing field '{key}'")
            if not isinstance(act[key], expected_type):
                raise ValueError(f"{act['id']}: field '{key}' has the wrong type")
        if act["id"] in seen_ids:
            raise ValueError(f"duplicate activity id: {act['id']}")
        seen_ids.add(act["id"])
        if act["cost_per_person"] < 0 or act["duration_hours"] <= 0:
            raise ValueError(f"{act['id']}: cost or duration out of range")
        if act["open_from"] >= act["open_until"]:
            raise ValueError(f"{act['id']}: open_from must be before open_until")
    return activities


def remaining_budget(conn, trip_id=1):
    """Total budget minus confirmed bookings minus itinerary costs."""
    row = conn.execute(
        """
        SELECT t.total_budget
               - COALESCE((SELECT SUM(cost) FROM booking
                           WHERE trip_id = t.id AND status = 'confirmed'), 0)
               - COALESCE((SELECT SUM(cost) FROM itinerary_item
                           WHERE trip_id = t.id), 0)
        FROM trip t WHERE t.id = ?
        """,
        (trip_id,),
    ).fetchone()
    return row[0]


def print_summary(db_path=DB_PATH):
    conn = sqlite3.connect(db_path)
    try:
        name, start, end, budget = conn.execute(
            "SELECT name, start_date, end_date, total_budget FROM trip WHERE id = 1"
        ).fetchone()
        members = conn.execute("SELECT COUNT(*) FROM member").fetchone()[0]
        confirmed = conn.execute(
            "SELECT COALESCE(SUM(cost), 0) FROM booking WHERE status = 'confirmed'"
        ).fetchone()[0]
        planned = conn.execute(
            "SELECT COALESCE(SUM(cost), 0) FROM itinerary_item"
        ).fetchone()[0]

        print(f"Trip: {name} | {start} to {end} | {members} members")
        print(f"  Total budget:        ${budget:,.2f}")
        print(f"  Confirmed bookings:  ${confirmed:,.2f}")
        print(f"  Itinerary costs:     ${planned:,.2f}")
        print(f"  Remaining budget:    ${remaining_budget(conn):,.2f}")
    finally:
        conn.close()


if __name__ == "__main__":
    create_database()
    activities = validate_activities()
    print(f"Created {DB_PATH.name} and validated {len(activities)} activities.\n")
    print_summary()