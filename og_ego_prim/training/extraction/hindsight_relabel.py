"""Hindsight Relabel — strip critic guidance prepended to chosen actor input.

Used by Track A only. Track B emits before guidance is prepended, so its
chosen rows are already clean.

Strategy: locate the literal anchor "Your input:" in the chosen actor's
input — everything before it is the prepended guidance and is dropped.
For maximum safety, the preferred relabel REPLACES chosen's input with
the rejected actor's input (which never had guidance prepended in the
first place). Strip-only is the fallback for edge cases where rejected
input is unavailable.
"""
from __future__ import annotations


YOUR_INPUT_ANCHOR = "Your input:"


def strip_guidance_from_input(input_text: str) -> str:
    """Drop everything before the 'Your input:' anchor.

    No-op if the anchor is at position 0 (already clean) or absent
    (caller must handle).
    """
    if not input_text:
        return input_text
    idx = input_text.find(YOUR_INPUT_ANCHOR)
    if idx <= 0:
        return input_text
    return input_text[idx:]


def relabel_chosen_input(chosen_input: str, rejected_input: str) -> str:
    """Replace chosen's input with rejected's (canonical Hindsight).

    Falls back to stripping prepended guidance from chosen if rejected is
    empty / missing.
    """
    if rejected_input:
        return rejected_input
    return strip_guidance_from_input(chosen_input)
