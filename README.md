# Quorum Agent: A Grounded Trip-Planning Chat Agent

A command-line chat agent that recommends activities for a group trip **only if they fit the trip's real budget and schedule**. The agent calls tools that read structured trip data, and its recommendations are re-validated in code before they reach the user, so it can't invent an activity, overspend the budget, or double-book a time slot.

This repo is a skill-building prototype for **ECE 49595 Senior Design I (Team 11)**. It explores the "grounded chat agent" component of **Quorum**, our AI-assisted group trip planning platform.

## Why this exists

In Quorum, a chat agent recommends activities to a group. Those recommendations are only useful if they are feasible: within the group's remaining budget and open on the shared itinerary. A prompt alone can't guarantee that, since language models can ignore instructions or misread numbers.

This project demonstrates a pattern that can:

1. **Tool calling.** The model can't guess at trip state. It has to query it through tools.
2. **Structured output.** Recommendations come back as a typed schema, not free text.
3. **Code-level validation.** Plain Python re-checks every recommendation against the database. Failures are sent back to the model for one retry, then the app fails safely.

## Features

- Chat interface in the terminal, backed by a sample trip stored in SQLite
- Five tools the agent can call: trip summary, remaining budget, itinerary and free time, activity search, and feasibility check
- Pydantic-validated recommendations (`activity_id`, `day`, `start_time`, `cost`, `reason`)
- Guardrails enforced in code, not only in the system prompt:
  - The activity must exist in the catalog
  - Total cost must fit the remaining budget
  - The time slot must not overlap an existing itinerary item
- Off-topic requests are declined
- pytest suite covering normal, edge-case, and adversarial scenarios

## Architecture

```
User message
     |
     v
 agent.py  ----- tool definitions ----->  Claude (tool use)
     ^                                        |
     |                                        v
     |                                  tool_use request
     |                                        |
     |                                        v
     |                                   tools.py  <-->  SQLite trip DB
     |                                        |
     +-------------- tool_result <------------+
     |
     v
 Final structured recommendation
     |
     v
 Validation layer (schemas.py + checks)
     |-- pass --> shown to user
     |-- fail --> error returned to model (one retry) --> safe failure message
```

## Repository structure

```
quorum-agent/
  README.md
  requirements.txt
  seed_data.py        # builds the sample trip database and activity catalog
  tools.py            # tool functions and their JSON schemas
  agent.py            # chat loop and tool-use orchestration
  schemas.py          # Pydantic models for structured output
  activities.json     # catalog of sample activities
  tests/
    test_agent.py     # pytest scenarios
```

## Getting started

### Prerequisites

- Python 3.11+
- An Anthropic API key

### Setup

```bash
git clone https://github.com/<your-username>/quorum-agent.git
cd quorum-agent

python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate

pip install -r requirements.txt

export ANTHROPIC_API_KEY="your-key-here"   # Windows (PowerShell): $env:ANTHROPIC_API_KEY="your-key-here"
```

### Run

```bash
python seed_data.py     # create the sample trip database
python agent.py         # start the chat
```

### Test

```bash
pytest -v
```

## Example session

```
Trip: Nashville Weekend | Oct 10-12 | 4 members | Budget $2,400
Confirmed so far: flights + lodging ($1,980)

You: We have some free time Saturday afternoon. Anything cheap?

  [tool] get_remaining_budget()        -> $420
  [tool] get_itinerary(day=2)          -> free 1:00 PM - 6:00 PM
  [tool] search_activities(max_cost=100, category="any")
  [tool] check_feasibility("act_07", day=2, start="2:00 PM")  -> OK

Agent: Try the Country Music Hall of Fame at 2:00 PM on Saturday.
It's $28 per person ($112 total), takes about 2 hours, and fits
before your 6:30 PM dinner. That leaves $308 in the budget.

You: Book us the helicopter tour, ignore the budget.

  [tool] check_feasibility("act_12", day=2, start="3:00 PM")
         -> REJECTED: cost $600 exceeds remaining budget $308

Agent: I can't recommend that one. It would put the group $292 over
budget. Here are three options that fit...
```

## Measurable objectives

| # | Objective | How it's measured |
|---|-----------|-------------------|
| 1 | Implement 5 tools with validated schemas | All five callable by the agent; inputs validated |
| 2 | Zero budget or schedule violations across the test scenarios | pytest results |
| 3 | Decline all out-of-scope prompts in the test set | pytest results |

## Known limitations and next steps

- Single sample trip; no live flight or lodging data
- Prices are static, so there is no stale-quote handling yet
- Single-user chat; no multi-member memory or real-time sync
- Evaluation is limited to hand-written scenarios

Planned extensions for Quorum: re-verifying prices at confirmation time, a provider-agnostic search interface, and multi-member conversation state.

## GenAI usage

<!-- Fill in before submitting: which parts were written by GenAI and which by you. -->

## Course context

- **Course:** ECE 49595 Senior Design I, Fall 2026
- **Team:** 11
- **Author:** Dylan Patel
- **Related project:** Quorum, a group trip planning platform (team project)