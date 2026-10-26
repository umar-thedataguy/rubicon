"""
mcp_server.py — expose Rubicon as a tool for AI clients over streamable HTTP.

Run:   python mcp_server.py
Then point Claude Desktop, Cursor, or any MCP client at http://127.0.0.1:8000/mcp
"""
try:                                   # mcp >= 2.0
    from mcp.server.mcpserver import MCPServer as Server
except ImportError:                    # mcp 1.x
    from mcp.server.fastmcp import FastMCP as Server

from rubicon.graph import run_pipeline

server = Server("rubicon")


@server.tool()
def audit_dataset(csv_path: str, target: str = "churn", treatment: str = "",
                  id_col: str = "", n_trials: int = 10) -> dict:
    """Audit a CSV for data leakage, repair it, run a 7-model tournament under the
    repaired evaluation plan, and (if a treatment column is given) estimate uplift.
    Returns the diagnosis, the evaluation plan, leaderboards and a plain-English
    explanation."""
    s = run_pipeline(csv_path, target, treatment or None, id_col or None,
                     compare_before_after=True, n_trials=n_trials)
    return {
        "diagnoses": [r.get("action") for r in s["rounds"] if r.get("action")],
        "evaluation_plan": s["plan"],
        "leaderboard_naive": s["baseline"]["leaderboard"],
        "leaderboard_after_audit": s["tournament"]["leaderboard"],
        "tuned": s["tournament"]["tuned"],
        "causal": {k: v for k, v in s.get("causal", {}).items() if k != "curves"},
        "explanation": s["explanation"]["text"],
        "llm": s["llm"],
        "logs": s["logs"],
    }


if __name__ == "__main__":
    server.run(transport="streamable-http")
