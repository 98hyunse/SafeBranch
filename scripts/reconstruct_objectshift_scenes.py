#!/usr/bin/env python3
"""Reconstruct ObjectShift scene templates from upstream templates and patches."""

import argparse
import hashlib
import json
from pathlib import Path


def canonical_bytes(value):
    return json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")


def digest(value):
    return hashlib.sha256(canonical_bytes(value)).hexdigest()


def decode_pointer(path):
    if not path.startswith("/"):
        raise ValueError(f"invalid JSON pointer: {path}")
    return [part.replace("~1", "/").replace("~0", "~") for part in path[1:].split("/")]


def apply_operations(document, operations):
    for operation in operations:
        parts = decode_pointer(operation["path"])
        parent = document
        for part in parts[:-1]:
            parent = parent[int(part)] if isinstance(parent, list) else parent[part]

        key = parts[-1]
        op = operation["op"]
        if isinstance(parent, list):
            index = len(parent) if key == "-" else int(key)
            if op == "add":
                parent.insert(index, operation["value"])
            elif op == "remove":
                parent.pop(index)
            elif op == "replace":
                parent[index] = operation["value"]
            else:
                raise ValueError(f"unsupported operation: {op}")
        elif op == "remove":
            del parent[key]
        elif op in {"add", "replace"}:
            parent[key] = operation["value"]
        else:
            raise ValueError(f"unsupported operation: {op}")
    return document


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--base-scenes-dir",
        type=Path,
        required=True,
        help="Upstream data/scenes directory containing the original templates.",
    )
    parser.add_argument(
        "--patch-dir",
        type=Path,
        default=Path("data/ood_object_shift/scene_patches"),
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    patch_files = sorted(args.patch_dir.glob("*.json"))
    if not patch_files:
        raise SystemExit(f"no patches found in {args.patch_dir}")

    for patch_file in patch_files:
        patch = json.loads(patch_file.read_text(encoding="utf-8"))
        base_path = args.base_scenes_dir / patch["base"]["relative_path"]
        base = json.loads(base_path.read_text(encoding="utf-8"))
        actual_base = digest(base)
        expected_base = patch["base"]["canonical_sha256"]
        if actual_base != expected_base:
            raise SystemExit(
                f"base checksum mismatch for {base_path}: "
                f"expected {expected_base}, got {actual_base}"
            )

        reconstructed = apply_operations(base, patch["operations"])
        actual_target = digest(reconstructed)
        expected_target = patch["target"]["canonical_sha256"]
        if actual_target != expected_target:
            raise SystemExit(
                f"reconstruction mismatch for {patch_file}: "
                f"expected {expected_target}, got {actual_target}"
            )

        output_path = args.output_dir / patch["target"]["relative_path"]
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(
            json.dumps(reconstructed, ensure_ascii=False, separators=(",", ":")),
            encoding="utf-8",
        )

    print(f"Reconstructed and verified {len(patch_files)} ObjectShift scene templates.")


if __name__ == "__main__":
    main()
