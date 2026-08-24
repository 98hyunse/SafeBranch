"""Canonical log writer for DFS runs.

Two entry points, one for today and one reserved for later:

1. :func:`save_canonical_for_task_dir` — **post-hoc helper**. Called at
   the end of a successful DFS run, right after ``_trace.json`` and
   ``report.json`` have been written. It re-parses those legacy files
   through :func:`og_ego_prim.harness.migrate.migrate_task` and saves a
   canonical ``sample.json`` beside them. Zero changes to planner
   execution flow, which keeps the blast radius small during active
   experiments.

2. :class:`CanonicalLogger` — **live observer**. Not wired into the
   planner yet. The shape below is a placeholder so future work can
   replace post-hoc migration without changing the CLI contract. The
   live path matters only for runs that crash mid-episode; for
   cleanly-terminating runs the post-hoc helper gives the same output.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

from .migrate import migrate_task
from .schema import Sample


DEFAULT_SAMPLE_FILENAME = "sample.json"


def save_canonical_for_task_dir(
    task_dir: str | Path,
    model: str,
    *,
    filename: str = DEFAULT_SAMPLE_FILENAME,
    runner_log: Optional[str | Path] = None,
) -> Path:
    """Materialise ``<task_dir>/sample.json`` from legacy artefacts.

    ``task_dir`` must already contain ``_trace.json`` / ``report.json``
    (the usual DFS output). If only partial artefacts exist the
    migrator will still produce a valid Sample with ``crash_info`` set,
    which is why this function is also safe to call from a crash-
    handler path later on.

    Returns the path the sample was written to.
    """
    task_dir_path = Path(task_dir)
    if not task_dir_path.is_dir():
        raise FileNotFoundError(f"task_dir does not exist: {task_dir_path}")

    runner_log_path = Path(runner_log) if runner_log else None
    sample: Sample = migrate_task(task_dir_path, model=model, runner_log=runner_log_path)

    output_path = task_dir_path / filename
    with output_path.open("w", encoding="utf-8") as f:
        f.write(sample.model_dump_json(indent=2, exclude_none=False))
    return output_path


def load_canonical_sample(task_dir: str | Path, filename: str = DEFAULT_SAMPLE_FILENAME) -> Optional[Sample]:
    """Read back a previously-saved canonical sample, if one exists."""
    path = Path(task_dir) / filename
    if not path.exists():
        return None
    with path.open("r", encoding="utf-8") as f:
        raw = f.read()
    return Sample.model_validate_json(raw)


# ---------------------------------------------------------------------------
# Placeholder for the future live-logging path
# ---------------------------------------------------------------------------


class CanonicalLogger:
    """**Not yet wired.** Reserved for the future live event-capture path.

    When implemented this will be handed to ``DFSPlanner`` as an
    optional observer; it will emit a canonical ``Sample`` that remains
    valid even if the planner crashes mid-run. Until then the post-hoc
    helper above is the supported entry point.

    The class exists so that CLI / orchestration code can refer to it
    by name ahead of the actual implementation without importers
    needing to know whether the post-hoc or live path is in use.
    """

    def __init__(self, task: str, scene: str, model: str, output_dir: str | Path):
        self.task = task
        self.scene = scene
        self.model = model
        self.output_dir = Path(output_dir)

    def finalize(self) -> Path:
        """Fallback: just run the post-hoc helper so the contract still
        yields a ``sample.json`` even if nothing was captured live."""
        return save_canonical_for_task_dir(self.output_dir, model=self.model)
