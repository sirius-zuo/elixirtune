from pathlib import Path


def _ws(domain: str) -> Path:
    """Return the workspace path for a given domain."""
    return Path("workspaces") / domain
