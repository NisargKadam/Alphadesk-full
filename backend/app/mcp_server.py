"""Session 04: the same desk tools, served over MCP.

MCP (Model Context Protocol) is the USB port for agent tools: any MCP client
— Claude, an IDE, another agent — can discover and call these functions with
zero AlphaDesk-specific code. One implementation, two protocols: the graph
calls these tools in-process; the outside world calls them over MCP.

Run standalone (stdio transport, which `claude mcp add` speaks):
    cd backend && .venv/bin/python app/mcp_server.py
"""

import sys
from pathlib import Path

# `claude mcp add` launches this file by path from an arbitrary cwd, so put
# backend/ on sys.path before importing the app package.
if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from mcp.server.fastmcp import FastMCP

from app.tools import get_price_history, get_quote

mcp = FastMCP("alphadesk")

# The exact functions the graph uses, unwrapped from their LangChain @tool
# shells (.func is the original callable) and re-registered with FastMCP —
# name, docstring, and type hints become the MCP tool schema, same as before.
mcp.tool()(get_quote.func)
mcp.tool()(get_price_history.func)

if __name__ == "__main__":
    mcp.run()  # stdio: JSON-RPC over stdin/stdout, one client per process
