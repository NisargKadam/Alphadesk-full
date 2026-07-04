# Session 06 — Multi-Agent Systems

## Learning objectives

- Teach the hand-off as a *typed contract*: `Task` goes out, `AgentResult` comes back — anything not in the schema does not survive the trip, by design.
- Build a supervisor that plans the MINIMAL set of delegations — selective delegation is clean design, fan-out is not free.
- Turn capabilities into specialists: scoped toolboxes (the news agent cannot quote prices) and one retrieval analyst that owns citations.
- Compose specialist results into one report without losing evidence — citations must survive two hops.
- Evolve, don't duplicate: the Session 02 router still triages; `desk` traffic now flows through the supervisor.

## Live-coding order

1. `backend/app/graph.py` — `Task` and `AgentResult` models first, before any agent exists. One why-comment carries the session: the hand-off is data with a schema, not a vibe.
2. `backend/app/graph.py` — `Plan` + `SUPERVISOR_PROMPT`; stress "MINIMAL set of tasks" and why (latency, cost, and blame all scale with fan-out).
3. `backend/app/graph.py` — `RouteDecision` narrows to `general | desk | unclear`; the router evolves into the supervisor's triage, not a duplicate of it.
4. `backend/app/graph.py` — the `supervisor` node: structured-output planning at temperature 0; each task emits a `handoff` event.
5. `backend/app/graph.py` — `run_tool_agent()`: a specialist is a narrow prompt + a *subset* of tools + the same bounded loop from Session 04. Tool events now carry `agent`.
6. `backend/app/graph.py` — the Market agent and News agent are one lambda each — the pattern, reused.
7. `backend/app/graph.py` — `run_filings_analyst()`: Session 03/05 retrieval wearing an agent's badge; evidence is numbered *here* and rides inside `AgentResult.evidence`.
8. `backend/app/graph.py` — the `agents` dispatcher: sequential on purpose (watch each lane fill); name LangGraph's `Send` API as the parallel production move.
9. `backend/app/graph.py` — the `report` node: compose results, *preserve* `[n]` markers, re-emit the analyst's evidence as the `citations` event — the survival test.
10. `backend/app/graph.py` — rewire: `desk → supervisor → agents → report`; `clarify → router` (a clarified message re-triages); `respond` slims back down to general chat — code that leaves is as teachable as code that arrives.
11. `backend/app/main.py` — the token filter gains `report` as a second answering node. First change to this file since Session 02; say that out loud.
12. Terminal — curl the briefing prompt; read the event stream: handoff ×3 → agent running/done ×3 → citations → tokens. **The delegation is legible on the wire.**

## Demo script

*Pre-class (playbook): ingest one real large-cap annual-report PDF so demo 1 spans live data and real citations.*

1. **"Brief me on Apple — price action this month, latest news, and risk factors from the filings."** — the supervisor fans out: three `handoff` events, then each specialist's `running → tool events → done` in sequence; one composed report with live prices *and* page citations. Read the `citations` payload: those numbers crossed two hand-offs intact.
2. **"What's Nvidia trading at?"** — the supervisor delegates to Market *only*. Selective delegation is clean design; point at the single `handoff`.
3. **"What does the annual report say about AI risk?"** — filings-only: one analyst delegation, citations intact through the report writer.

## Where it breaks

- **The report has facts but no `[n]` markers, or renumbered ones.** The classic multi-agent failure: evidence lost between agents. The fix is structural, not prompt-side — citations ride inside `AgentResult.evidence` with their numbers fixed at the analyst, and the report writer is *forbidden* from renumbering. If markers vanish, check the report prompt still says "keep every [n] exactly as written".
- **Every question fans out to all three agents.** The supervisor prompt lost its "MINIMAL set" teeth, or planning is running at a creative temperature. Plan at temperature 0 and demo 2 becomes reproducible.
- **Tool events arrive without their specialist.** The `agent` field is missing — the specialist's `run_tool_agent` writer calls must include `"agent": name`; that string is how any client (including the final UI) attributes a call to its lane.
- **The briefing takes ~30s of quiet tokens-free stream.** Three sequential specialists each make model calls; the `agent running/done` events exist precisely so the wait reads as work. If it genuinely hangs, one specialist's tool loop isn't terminating — check the bounded-loop guard.

## Stretch

> **Budget note:** ~332 changed application lines against the ~250 target. The named trim: the **News agent** — it is one lambda plus one prompt line, so cut it live and teach it back in 90 seconds; Market + Filings demonstrate the full pattern.

- Parallelize the `agents` node with LangGraph's `Send` API and watch the lanes fill concurrently — same contracts, different scheduler.
- Add a `sources_used` field to `AgentResult` and have the report writer print an attribution line per specialist.
- Give the supervisor a `none` option (answer directly without specialists) and measure how often it takes it — delegation should be earned, not default.
