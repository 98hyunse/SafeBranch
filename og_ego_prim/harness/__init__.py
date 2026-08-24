"""IS-Bench research harness: canonical logging, migration, and analysis.

Layer boundaries:
- schema.py: canonical data models (source of truth for log format)
- migrate.py: recovery tool for legacy results_dfs_*/ runs
- logger.py: post-hoc sample.json writer + future live Logger
- loader.py: read canonical logs back into memory             [TBD]
"""

from .logger import (
    CanonicalLogger,
    DEFAULT_SAMPLE_FILENAME,
    load_canonical_sample,
    save_canonical_for_task_dir,
)
from .migrate import migrate_run, migrate_task
from .schema import (
    SCHEMA_VERSION,
    ExperimentLog,
    Sample,
    Identity,
    Config,
    Status,
    Evaluation,
    ProcessSafetyCheck,
    TerminationSafetyCheck,
    ExecutionGoal,
    AwarenessRecord,
    FinalStateSummary,
    Artifacts,
    ExternalRefs,
    CrashInfo,
    Timestamps,
    Event,
    ExecuteEvent,
    TerminalEvent,
    ActorFailureEvent,
    CriticEvent,
    BacktrackEvent,
    UnsafeAcceptedEvent,
    GuidanceUpdateEvent,
    SpanBeginEvent,
    SpanEndEvent,
    ErrorEvent,
    Reflection,
    HazardOrigin,
    RepairPoint,
)

__all__ = [
    "CanonicalLogger",
    "DEFAULT_SAMPLE_FILENAME",
    "load_canonical_sample",
    "save_canonical_for_task_dir",
    "migrate_run",
    "migrate_task",
    "SCHEMA_VERSION",
    "ExperimentLog",
    "Sample",
    "Identity",
    "Config",
    "Status",
    "Evaluation",
    "ProcessSafetyCheck",
    "TerminationSafetyCheck",
    "ExecutionGoal",
    "AwarenessRecord",
    "FinalStateSummary",
    "Artifacts",
    "ExternalRefs",
    "CrashInfo",
    "Timestamps",
    "Event",
    "ExecuteEvent",
    "TerminalEvent",
    "ActorFailureEvent",
    "CriticEvent",
    "BacktrackEvent",
    "UnsafeAcceptedEvent",
    "GuidanceUpdateEvent",
    "SpanBeginEvent",
    "SpanEndEvent",
    "ErrorEvent",
    "Reflection",
    "HazardOrigin",
    "RepairPoint",
]
