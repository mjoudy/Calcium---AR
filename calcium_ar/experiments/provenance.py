"""
Record which version of the code produced a run.

For results to be reproducible "for later users", every run must be tied to the
exact commit that generated it. ``git_revision()`` captures that commit hash plus
a flag for whether the working tree had uncommitted changes to tracked files
(``git_dirty=True`` means the code differed from any commit, so the result is not
cleanly reproducible from git alone — re-run from a clean, committed checkout).
"""

from __future__ import annotations

import subprocess
from pathlib import Path


def git_revision(repo_dir: str | Path | None = None) -> dict:
    """Return ``{'git_commit': <hash>, 'git_dirty': <bool>}`` for the repo that
    holds this package.

    Falls back to ``{'unknown', False}`` if git is unavailable or this is not a
    git checkout (e.g. an installed wheel), so logging a run never fails because
    of provenance capture.
    """
    repo_dir = Path(repo_dir) if repo_dir is not None else Path(__file__).resolve().parent

    def _git(*args: str) -> str:
        return subprocess.check_output(
            ["git", *args], cwd=repo_dir, stderr=subprocess.DEVNULL
        ).decode().strip()

    try:
        commit = _git("rev-parse", "HEAD")
        dirty = bool(_git("status", "--porcelain", "--untracked-files=no"))
        return {"git_commit": commit, "git_dirty": dirty}
    except Exception:
        return {"git_commit": "unknown", "git_dirty": False}
