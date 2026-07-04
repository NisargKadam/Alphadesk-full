# Session 04 — Tool Use, Function Calling & MCP

## Learning objectives

- Understand function calling as a protocol: the model *requests* calls against schemas you publish; executing them is your job.
- Build the tool loop by hand: respond → tools → respond, until the model answers in plain text.
- Write tools that fail gracefully — a typed error dict instead of an exception, so one bad ticker can't kill a graph run.
- Serve the identical functions over MCP and watch an external agent (Claude) call them — one implementation, two protocols.
- Extend the visible pipeline again: `tool_call` / `tool_result` land on the same custom stream channel as `node` and `citations`.

## Live-coding order

1. Terminal — `uv pip install yfinance mcp`; re-pin `requirements.txt`.
2. `backend/app/tools.py` — `get_quote` with the `@tool` decorator; the docstring and type hints ARE the interface the model reads.
3. `backend/app/tools.py` — the try/except returning `{"error": ...}`; discuss why tools must fail gracefully (an exception kills the run; an error payload lets the model route around it).
4. `backend/app/tools.py` — `get_price_history` (downsample to ≤30 closes) and `get_company_news` (defensive field access — Yahoo's schema shifts under yfinance).
5. Terminal — smoke-test the tools in a REPL: `get_quote.invoke({"ticker": "TSLA"})`. **Live data, ~20 minutes in.**
6. `backend/app/graph.py` — `bind_tools(MARKET_TOOLS)` on the market route; the request now carries JSON schemas.
7. `backend/app/graph.py` — the `tools` node, hand-rolled: execute each requested call, wrap results in `ToolMessage`s keyed by `tool_call_id`. (`langgraph.prebuilt.ToolNode` does this for you; note that old tutorials' `langgraph.prebuilt.create_react_agent` is deprecated in favor of `langchain.agents.create_agent`.)
8. `backend/app/graph.py` — the conditional loop: `respond → tools` while the last message has `tool_calls`, else `END`; `tools → respond` closes the circle.
9. `backend/app/graph.py` — emit `tool_call` / `tool_result` on the custom stream channel, same as `node` and `citations`. `main.py`: still untouched since Session 02.
10. Terminal — curl "What's Tesla trading at right now?"; read the event sequence out loud: router → respond → tool_call → tool_result → respond → tokens.
11. `backend/app/mcp_server.py` — FastMCP; re-register the same functions (`.func` unwraps the LangChain shell); stdio transport.
12. Terminal — `claude mcp add alphadesk -- <venv-python> <path>/app/mcp_server.py`, then in Claude: "use alphadesk to quote TSLA".

## Demo script

1. **"What's Tesla trading at right now?"** — watch `tool_call` fire with the schema-shaped args, then `tool_result` carry live numbers. Read the path out loud: `router → respond → tools → respond` — the loop is visible on the wire.
2. **"Compare Apple and Microsoft over the last month."** — the model chooses *two* `get_price_history` calls on its own; two result payloads, then a comparison in prose.
3. **"Any recent news on Nvidia?"** — a headlines payload. Then switch to Claude and call the same tools over MCP — the desk's functions, consumed by an agent we didn't write.

## Where it breaks

- **`KeyError: 'lastPrice'` or every quote errors.** yfinance's `fast_info` keys and news schema drift between releases. Pin the version from `requirements.txt`, and treat this as the live case study for why the tools return error dicts instead of raising.
- **The model answers from memory and never calls a tool.** `bind_tools` wasn't applied on the market route (or the router sent the message to `general`). No schema in the request → nothing for the model to call.
- **The graph loops forever calling tools.** The conditional edge checks the *last* message's `tool_calls`; if the `ToolMessage`s aren't keyed with the right `tool_call_id`, the model never sees its answers and asks again.
- **MCP server "works in the terminal, not in Claude".** `claude mcp add` launches the file from an arbitrary cwd — that's what the `sys.path` shim at the top of `mcp_server.py` is for. Use absolute paths in the config.

## Stretch

> **Budget note:** ~285 changed application lines against the ~250 target. `get_company_news` (~35 lines) is the self-contained cut if trimming live — teach it here instead.

- Expose `search_filings` over MCP too, and ask Claude to answer a SampleCo question through your retrieval stack — RAG-as-a-service in one line.
- Add `get_quote`'s 52-week range to the payload and let the model reason about where today's price sits in it.
- Return a `stale: true` flag when yfinance serves cached data after hours; watch the model start hedging its phrasing — tools shape tone.
