"""exp10 conversation md parser primitives.

Section header patterns (locked from real exp10 data):
  ## [N] Actor
  ## [N] Actor (with guidance)
  ## [N] BeforeBDDLReflector
  ## [N] TaskFailReflector
  ## [N] TermSafetyReflector
  ## [N] · ★ EXEC_COMMITTED
  ## [N] · DEEP_BACKTRACK
  ## [N] · EXEC_FAILED ...

Body shape per LLM-call section:
  **Input** (post-system-prompt section only):
  ```
  ...
  ```
  **Response:**
  ```
  ```json
  ...
  ```
  ```

Marker sections carry only `- **step**: N`, `- **action**: ...`, `- **reasoning**: ...`.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any, List, Optional


SECTION_RE = re.compile(r"^## \[(\d+)\] (.+?)\s*$", re.M)
INPUT_BLOCK_RE = re.compile(
    r"\*\*Input\*\*\s*(?:\([^)]*\))?\s*:\s*\n```\s*\n(.*?)\n```",
    re.S,
)
RESPONSE_BLOCK_RE = re.compile(r"\*\*Response:\*\*\s*\n```\s*\n(.*?)\n```", re.S)
JSON_FENCE_RE = re.compile(r"```json\s*\n(.*?)\n```", re.S)
# Handles the case where the outer RESPONSE_BLOCK_RE consumed the closing ```,
# leaving response_text as "```json\n{...}" with no closing fence.
JSON_OPEN_RE = re.compile(r"```json\s*\n(.*)", re.S)
MARKER_KV_RE = re.compile(r"^- \*\*(\w+)\*\*:\s*(.+?)\s*$", re.M)

CRITIC_LABELS = {"BeforeBDDLReflector", "TaskFailReflector", "TermSafetyReflector"}
MARKER_PATTERNS = (
    "★ EXEC_COMMITTED",
    "DEEP_BACKTRACK",
    "EXEC_FAILED",
    "EXEC_LOOP",
    "PHASE3_TASK_GOAL_FAIL",
    "TERM_SAFETY_FAIL",
)


@dataclass
class Section:
    index: int                                  # the [N] number
    label: str                                  # raw label after [N]
    kind: str                                   # "actor" | "critic" | "marker"
    input_text: Optional[str] = None
    response_text: Optional[str] = None
    response_json: Optional[dict] = None
    marker_data: dict = field(default_factory=dict)


def _classify(label: str) -> str:
    label_strip = label.lstrip("· ").strip()
    if label.startswith("Actor"):
        return "actor"
    if label_strip in CRITIC_LABELS:
        return "critic"
    if any(pat in label_strip for pat in MARKER_PATTERNS):
        return "marker"
    if label_strip.startswith("·"):
        return "marker"
    if "·" in label:
        return "marker"
    return "marker"


def _parse_marker_body(body: str) -> dict:
    out: dict[str, Any] = {}
    for m in MARKER_KV_RE.finditer(body):
        key, val = m.group(1), m.group(2).strip()
        if key == "step":
            try:
                out[key] = int(val)
            except ValueError:
                out[key] = val
        else:
            out[key] = val
    return out


def _parse_llm_body(body: str) -> tuple[Optional[str], Optional[str], Optional[dict]]:
    im = INPUT_BLOCK_RE.search(body)
    rm = RESPONSE_BLOCK_RE.search(body)
    input_text = im.group(1) if im else None
    response_text = rm.group(1) if rm else None
    response_json: Optional[dict] = None
    if response_text:
        jm = JSON_FENCE_RE.search(response_text)
        if jm:
            payload = jm.group(1)
        else:
            # Outer fence consumed the closing ```, try open-ended match
            jm2 = JSON_OPEN_RE.search(response_text)
            payload = jm2.group(1).strip() if jm2 else response_text
        try:
            response_json = json.loads(payload)
        except json.JSONDecodeError:
            response_json = None
    return input_text, response_text, response_json


def parse_conversation_md(md_text: str) -> List[Section]:
    sections: List[Section] = []
    matches = list(SECTION_RE.finditer(md_text))
    for i, m in enumerate(matches):
        start = m.start()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(md_text)
        body = md_text[start:end]
        index = int(m.group(1))
        label = m.group(2).strip()
        kind = _classify(label)
        if kind == "marker":
            sections.append(Section(
                index=index,
                label=label.lstrip("· ").strip("★ "),
                kind=kind,
                marker_data=_parse_marker_body(body),
            ))
        else:
            input_text, response_text, response_json = _parse_llm_body(body)
            sections.append(Section(
                index=index,
                label=label,
                kind=kind,
                input_text=input_text,
                response_text=response_text,
                response_json=response_json,
            ))
    return sections
