"""
Model clients / agents.

Keep imports lightweight to avoid forcing heavy simulator dependencies
when only a text-only client is needed (e.g., dataset generation CLIs).
"""

from .server_inference import ServerClient

try:
    from .plan_agent import PlanningAgent
except Exception:  # pragma: no cover
    PlanningAgent = None

__all__ = ["ServerClient", "PlanningAgent"]