# AlphaDesk — Architecture

The finished system, ten sessions later. Every layer below was added by one
session and is visible on the wire as SSE events (see "Streaming protocol"
in the README).

```mermaid
flowchart TB
    CLIENT["client (curl / UI)<br/>SSE reader"]

    subgraph API["FastAPI — backend/app/main.py"]
        CHAT["POST /api/chat · /api/chat/resume"]
        HEALTH["GET /health"]
    end

    subgraph GRAPH["LangGraph StateGraph — graph.py"]
        MEM["memory (S07)"] --> GUARD["guard (S08)"]
        GUARD --> ROUTER["router (S02)"]
        ROUTER -->|general| RESPOND["respond"]
        ROUTER -->|unclear| CLARIFY["clarify · HITL interrupt"]
        CLARIFY --> ROUTER
        ROUTER -->|desk| SUP["supervisor (S06)"]
        SUP --> AGENTS["agents"]
        AGENTS --> MARKET["market"]
        AGENTS --> NEWS["news"]
        AGENTS --> FIL["filings analyst (S03/S05)"]
        AGENTS --> REPORT["report"]
    end

    SQLITE[("desk_state.db<br/>checkpoints + profiles (S07)")]
    CHROMA[("chroma_db<br/>parent-child index (S03/S05)")]
    OPENAI["OpenAI API"]
    YF["yfinance (S04)"]
    OPIK["local Opik (S09)"]
    MCPS["app/mcp_server.py<br/>same tools over MCP (S04)"]

    CLIENT --> CHAT
    CHAT --> GRAPH
    GRAPH --> SQLITE
    FIL --> CHROMA
    MARKET --> YF
    NEWS --> YF
    GRAPH --> OPENAI
    CHAT -.->|OpikTracer (env-gated)| OPIK
    MCPS --> YF
```

## Deployment shape (Session 10)

One container (backend/Dockerfile), one named volume mounted at `/state`
(`DESK_DB_PATH`, `CHROMA_DIR` point into it), filings read-only at `/data`.
`GET /health` is both the human probe and the compose healthcheck. The UI
ships after the curriculum as a static bundle — the backend is the product
of the course.
