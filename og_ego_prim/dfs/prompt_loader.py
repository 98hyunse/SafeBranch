# -*- coding: utf-8 -*-
"""
Loader for DFS prompt YAML files.

Prompts are stored under og_ego_prim/dfs/prompts/<component>/<version>.yaml.
Each YAML file is a plain dict whose keys depend on the component:
  actor        → prompt
  prm          → system, user
  before_bddl  → system, user, feedback_block
  task_fail    → system, user, feedback_block
  term_safety  → system, user, feedback_block
"""

import os
from typing import Dict

import yaml

_PROMPTS_DIR = os.path.join(os.path.dirname(__file__), "prompts")


def load_dfs_prompt(component: str, version: str) -> Dict[str, str]:
    """Load a DFS prompt YAML file and return its contents as a dict.

    Args:
        component: Prompt component name (actor, prm, before_bddl, task_fail, term_safety).
        version:   Version string, e.g. 'v0', 'v1'.

    Returns:
        Dict mapping YAML keys to prompt strings.

    Raises:
        FileNotFoundError: If the requested component/version YAML does not exist.
    """
    path = os.path.join(_PROMPTS_DIR, component, f"{version}.yaml")
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"DFS prompt not found: {component}/{version}.yaml  (looked in {_PROMPTS_DIR})"
        )
    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    if not isinstance(data, dict):
        raise ValueError(f"DFS prompt file must be a YAML mapping: {path}")
    return data
