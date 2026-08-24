# Reproduction scope

This repository includes the minimum SafeBranch code path for collection,
same-state branch extraction, pair filtering, training-data conversion, and
ID/OOD evaluation. It intentionally excludes collected branch pairs, rollout
images, checkpoints, adapters, logs, and unrelated experimental tooling.

## Upstream requirements

Install OmniGibson 1.1.1, BDDL, and the BEHAVIOR assets separately under their
upstream terms. Activate that environment before using the entrypoint, then
install the lightweight Python dependencies in `requirements.txt`.

The 32-task ID manifest is included, but the corresponding base IS-Bench task,
BDDL, and scene files must be obtained from the upstream benchmark. This
release only redistributes the two newly constructed OOD benchmark packages.

## OOD evaluation

ObjectShift requires reconstructed scene templates:

```bash
python scripts/reconstruct_objectshift_scenes.py \
  --base-scenes-dir /path/to/upstream/data/scenes \
  --patch-dir data/ood_object_shift/scene_patches \
  --output-dir /path/to/objectshift/scenes
```

Load the common paper defaults and one evaluation condition, then provide the
actor endpoint and dataset paths explicitly:

```bash
set -a
source configs/evaluation/paper_defaults.env
source configs/evaluation/critic_free.env
set +a

MODEL_NAME_OR_PATH=<served-model-or-adapter> \
SERVER_IP=http://127.0.0.1:8000/v1 \
TASK_CONFIG_DIR="$PWD/data/ood_task_shift/tasks" \
BDDL_DIR="$PWD/data/ood_task_shift/bddl" \
OUTPUT_DIR="$PWD/results/task_shift" \
bash entrypoints/dfs_collect.sh 2 data/ood_task_shift/paper_eval_138.txt
```

For ObjectShift, use `data/ood_object_shift/tasks`,
`data/ood_object_shift/bddl`, and the reconstructed directory as `SCENE_DIR`.
The four condition files reproduce actor-only, critic-free, critic-full, and
critic-full-without-task-failure-recovery toggles. Actor-only and critic-free
have identical critic toggles; they differ in whether the served actor is the
base model or the trained SafeBranch adapter.

Summarize a completed result directory with:

```bash
python -m og_ego_prim.cli.summary_dfs_canonical \
  --result_dir /path/to/results
```

## Branch collection and training-data preparation

The paper pipeline is intentionally exposed as explicit stages:

1. Collect rollback branches with `entrypoints/dfs_collect.sh`.
2. Extract emitted branches with
   `python -m og_ego_prim.training.extraction.cli.extract_branches_b`.
3. Merge temperature pools with `scripts/merge_branch_pools.py`.
4. Judge and filter pairs with `scripts/quality_llm_judge.py` and
   `scripts/llm_judge_filter.py`.
5. Create the risk-stratified split with
   `scripts/train_test_split_by_risk.py` (seed `20260515`).
6. Project the train split to SFT and preference views with
   `scripts/split_to_sft_dpo.py`.
7. Convert the result to training-server paths with
   `scripts/prepare_training_data.py`.

SafeBranch's BranchPO stage uses the standard DPO objective on the extracted
same-state branch pairs; the contribution is pair construction rather than a
modified optimizer. Recorded paper hyperparameters are in
`configs/training/paper_hyperparameters.yaml`.

The original LLaMA-Factory YAML files were not available on this host during
release staging. The framework-neutral values above are verified from the
project runbooks, but they are not represented as an exact original training
command until those YAML files are recovered and compared.
