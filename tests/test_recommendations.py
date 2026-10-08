"""Offline coverage for the Exercise 5 recommendation extension."""

import sqlite3

import pytest

import tools
from test_agent import FakeBlock, FakeClient, FakeResponse, _propose_call, _text_reply
import agent


def ids(result):
    return [option["activity_id"] for option in result["recommendations"]]


def test_ranked_options_include_only_feasible_activities():
    result = tools.recommend_activities(2, "14:00")
    assert result["remaining_budget"] == 420
    assert result["count"] == 10
    assert ids(result) == ["act_01", "act_02", "act_03", "act_05", "act_07", "act_09", "act_08", "act_15", "act_04", "act_06"]
    museum = next(r for r in result["recommendations"] if r["activity_id"] == "act_07")
    assert (museum["total_cost"], museum["end_time"], museum["remaining_after"]) == (112, "16:00", 308)
    assert tools.get_remaining_budget()["remaining_budget"] == 420  # read-only


def test_maximum_cost_is_for_the_group_and_inclusive():
    result = tools.recommend_activities(2, "14:00", max_cost=24)
    assert ids(result) == ["act_01", "act_02", "act_03"]
    assert ids(tools.recommend_activities(2, "14:00", max_cost=0)) == ["act_01"]


def test_user_limit_cannot_override_remaining_budget():
    result = tools.recommend_activities(2, "14:00", max_cost=10000)
    assert result["cost_limit"] == 420
    assert "act_12" not in ids(result)


def test_category_filter_and_empty_result():
    assert ids(tools.recommend_activities(2, "14:00", category="museum")) == ["act_07"]
    result = tools.recommend_activities(2, "14:00", category="museum", max_cost=100)
    assert result["count"] == 0
    assert result["recommendations"] == []
    assert "No activities" in result["message"]
    assert tools.recommend_activities(2, "14:00", category="unknown")["count"] == 0


def test_schedule_conflicts_and_open_hours_remove_options():
    assert tools.recommend_activities(2, "19:00")["count"] == 0  # dinner conflicts
    assert "act_10" not in ids(tools.recommend_activities(2, "14:00"))  # opens 19:00
    assert ids(tools.recommend_activities(2, "20:30", category="tour")) == ["act_10"]
    # Ending exactly when dinner begins is permitted.
    assert "act_03" in ids(tools.recommend_activities(2, "18:00", category="food"))


@pytest.mark.parametrize("day,time", [(0,"14:00"),(4,"14:00"),(True,"14:00"),(1.5,"14:00"),("2","14:00"),(2,"2pm"),(2,"9:00"),(2,"24:00"),(2,"14:60"),(2,None)])
def test_invalid_slot_is_reported_without_crashing(day, time):
    assert "error" in tools.recommend_activities(day, time)
    assert tools.check_feasibility("act_07", day, time)["feasible"] is False


@pytest.mark.parametrize("cost", [-1, float("nan"), float("inf"), True, "100"])
def test_invalid_cost_limit_is_reported(cost):
    assert "error" in tools.recommend_activities(2, "14:00", max_cost=cost)


@pytest.mark.parametrize("category", ["", "   ", 42])
def test_invalid_category_is_reported(category):
    assert "error" in tools.recommend_activities(2, "14:00", category=category)


def test_missing_trip_is_reported():
    assert "no trip" in tools.recommend_activities(2, "14:00", trip_id=999)["error"]


def test_past_midnight_is_rejected_instead_of_clamped():
    result = tools.check_feasibility("act_11", 1, "22:00")
    assert result["feasible"] is False
    assert "same-day boundary" in result["reason"]
    assert "act_11" not in ids(tools.recommend_activities(1, "22:00"))


def test_equal_cost_options_sort_by_name(monkeypatch):
    catalog = tools._load_activities()
    alpha = {**catalog["act_01"], "id": "alpha", "name": "Alpha Walk"}
    zulu = {**alpha, "id": "zulu", "name": "Zulu Walk"}
    monkeypatch.setattr(tools, "_load_activities", lambda: {"zulu": zulu, "alpha": alpha})
    assert ids(tools.recommend_activities(2, "14:00")) == ["alpha", "zulu"]


def test_recommendations_recompute_changed_budget_group_and_schedule():
    assert "act_07" in ids(tools.recommend_activities(2, "14:00"))
    with sqlite3.connect(tools.DB_PATH) as conn:
        conn.execute("UPDATE trip SET total_budget = 2080 WHERE id = 1")  # $100 left
    assert "act_07" not in ids(tools.recommend_activities(2, "14:00"))
    with sqlite3.connect(tools.DB_PATH) as conn:
        conn.execute("DELETE FROM member WHERE id = 4")
    result = tools.recommend_activities(2, "14:00")
    assert result["group_size"] == 3
    assert "act_07" in ids(result)  # $84 now fits
    with sqlite3.connect(tools.DB_PATH) as conn:
        conn.execute("INSERT INTO itinerary_item VALUES (99, 1, 2, '14:00', '15:00', 'New meeting', 0)")
    assert tools.recommend_activities(2, "14:00")["count"] == 0


def test_trip_without_members_is_rejected():
    with sqlite3.connect(tools.DB_PATH) as conn:
        conn.execute("DELETE FROM member WHERE trip_id = 1")
    assert "member" in tools.recommend_activities(2, "14:00")["error"]


def test_tool_schema_and_dispatcher():
    definition = next(t for t in tools.TOOL_DEFINITIONS if t["name"] == "recommend_activities")
    assert definition["input_schema"]["required"] == ["day", "start_time"]
    assert ids(tools.call_tool("recommend_activities", {"day": 2, "start_time": "14:00", "category": "museum"})) == ["act_07"]
    assert "error" in tools.call_tool("recommend_activities", {"day": 2})


def test_agent_can_discover_options_then_validate_a_proposal():
    client = FakeClient([
        FakeResponse([FakeBlock(type="tool_use", id="discover", name="recommend_activities", input={"day": 2, "start_time": "14:00", "category": "museum"})], "tool_use"),
        _propose_call("proposal", "act_07", 2, "14:00"),
        _text_reply("The museum fits your afternoon."),
    ])
    messages = [{"role": "user", "content": "Find a museum for Saturday afternoon."}]
    assert "museum" in agent.run_turn(client, messages)
    assert client.calls == 3
    assert "recommend_activities" in client.tool_names_per_call[0]
    assert "act_07" in messages[2]["content"][0]["content"]
    assert messages[4]["content"][0]["is_error"] is False
