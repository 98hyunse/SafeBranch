#!/usr/bin/env python3
"""Build the public OOD benchmark package from an IS-Bench working tree."""

import argparse
import hashlib
import json
import shutil
from pathlib import Path


def file_sha256(path):
    hasher = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def canonical_sha256(value):
    raw = json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def escape_pointer(part):
    return str(part).replace("~", "~0").replace("/", "~1")


def make_patch(base, target, path=""):
    if type(base) is not type(target):
        return [{"op": "replace", "path": path, "value": target}]
    if isinstance(base, dict):
        operations = []
        for key in sorted(base.keys() - target.keys()):
            operations.append({"op": "remove", "path": f"{path}/{escape_pointer(key)}"})
        for key in sorted(target.keys() - base.keys()):
            operations.append(
                {"op": "add", "path": f"{path}/{escape_pointer(key)}", "value": target[key]}
            )
        for key in sorted(base.keys() & target.keys()):
            operations.extend(
                make_patch(base[key], target[key], f"{path}/{escape_pointer(key)}")
            )
        return operations
    if isinstance(base, list):
        return [] if base == target else [{"op": "replace", "path": path, "value": target}]
    return [] if base == target else [{"op": "replace", "path": path, "value": target}]


def copy_bddl(source_root, task_names, destination):
    for task_name in task_names:
        source = source_root / "data" / "bddl" / task_name / "problem0.bddl"
        if not source.is_file():
            raise FileNotFoundError(source)
        target = destination / task_name / "problem0.bddl"
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)


def copy_tasks(source_dir, task_names, destination):
    destination.mkdir(parents=True, exist_ok=True)
    for task_name in task_names:
        source = source_dir / f"{task_name}.json"
        if not source.is_file():
            raise FileNotFoundError(source)
        shutil.copy2(source, destination / source.name)


def write_checksums(dataset_dir):
    files = sorted(
        path for path in dataset_dir.rglob("*")
        if path.is_file() and path.name != "SHA256SUMS"
    )
    lines = [f"{file_sha256(path)}  {path.relative_to(dataset_dir)}" for path in files]
    (dataset_dir / "SHA256SUMS").write_text("\n".join(lines) + "\n", encoding="utf-8")


def build_object_shift(source_root, output_root):
    output = output_root / "ood_object_shift"
    source_manifest = json.loads(
        (source_root / "data/tasks_typeA/_meta/manifest.json").read_text(encoding="utf-8")
    )
    names = [entry["variant_name"] for entry in source_manifest]
    if len(names) != 147 or len(set(names)) != 147:
        raise ValueError("ObjectShift manifest must contain 147 unique variants")

    copy_tasks(source_root / "data/tasks_typeA", names, output / "tasks")
    copy_bddl(source_root, names, output / "bddl")
    (output / "task_list.txt").write_text("\n".join(names) + "\n", encoding="utf-8")
    (output / "source_manifest.json").write_text(
        json.dumps(source_manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )

    patch_dir = output / "scene_patches"
    patch_dir.mkdir(parents=True, exist_ok=True)
    for entry in source_manifest:
        scene = entry["scene"]
        task_name = entry["task_id"]
        variant = entry["variant_name"]
        base_rel = Path(scene) / "json" / f"{scene}_task_{task_name}_0_0_template.json"
        target_rel = Path(scene) / "json" / f"{scene}_task_{variant}_0_0_template.json"
        base_path = source_root / "data/scenes" / base_rel
        target_path = source_root / "data/scenes" / target_rel
        base = json.loads(base_path.read_text(encoding="utf-8"))
        target = json.loads(target_path.read_text(encoding="utf-8"))
        patch = {
            "format": "safebranch-json-patch-v1",
            "base": {
                "relative_path": str(base_rel),
                "canonical_sha256": canonical_sha256(base),
            },
            "target": {
                "relative_path": str(target_rel),
                "canonical_sha256": canonical_sha256(target),
            },
            "operations": make_patch(base, target),
        }
        (patch_dir / f"{variant}.json").write_text(
            json.dumps(patch, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )

    manifest = {
        "name": "SafeBranch OOD-ObjectShift",
        "version": 1,
        "num_variants": len(names),
        "task_list": "task_list.txt",
        "tasks_dir": "tasks",
        "bddl_dir": "bddl",
        "scene_patches_dir": "scene_patches",
        "requires_upstream_scene_templates": True,
        "contains_3d_assets": False,
    }
    (output / "manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    write_checksums(output)


def build_task_shift(source_root, output_root):
    output = output_root / "ood_task_shift"
    full_source = source_root / "_workspace/eval_baselines_2026/lists/newtask_159.txt"
    paper_source = source_root / "_workspace/intersection_local.txt"
    full = [
        line.strip() for line in full_source.read_text(encoding="utf-8").splitlines()
        if line.strip() and line.strip() != "_report"
    ]
    paper = [line.strip() for line in paper_source.read_text(encoding="utf-8").splitlines() if line.strip()]
    if len(full) != 159 or len(set(full)) != 159:
        raise ValueError("TaskShift full manifest must contain 159 unique tasks")
    if len(paper) != 138 or len(set(paper)) != 138 or not set(paper) <= set(full):
        raise ValueError("TaskShift paper manifest must be a 138-task subset of the full set")

    copy_tasks(source_root / "data/tasks_method3_syth", full, output / "tasks")
    copy_bddl(source_root, full, output / "bddl")
    (output / "full_159.txt").write_text("\n".join(full) + "\n", encoding="utf-8")
    (output / "paper_eval_138.txt").write_text("\n".join(paper) + "\n", encoding="utf-8")
    manifest = {
        "name": "SafeBranch OOD-TaskShift",
        "version": 1,
        "num_generated_variants": len(full),
        "num_paper_evaluation_variants": len(paper),
        "full_task_list": "full_159.txt",
        "paper_evaluation_task_list": "paper_eval_138.txt",
        "tasks_dir": "tasks",
        "bddl_dir": "bddl",
        "contains_3d_assets": False,
    }
    (output / "manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    write_checksums(output)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    args.output_root.mkdir(parents=True, exist_ok=True)
    build_object_shift(args.source_root.resolve(), args.output_root.resolve())
    build_task_shift(args.source_root.resolve(), args.output_root.resolve())
    print("Built ObjectShift (147) and TaskShift (159 full / 138 paper evaluation).")


if __name__ == "__main__":
    main()
