# OOD-ObjectShift

OOD-ObjectShift contains 147 task variants that add either a neutral object or
a hazard-associated distractor while preserving the original task goal.

## Contents

- `tasks/`: SafeBranch task configuration JSON files.
- `bddl/`: BDDL problem definitions.
- `task_list.txt`: all 147 evaluation task names.
- `source_manifest.json`: variant construction metadata.
- `scene_patches/`: changes against upstream scene templates. These patches do
  not contain 3D assets.

## Reconstruct scene templates

After obtaining the upstream OmniGibson/BEHAVIOR scene templates, run:

```bash
python scripts/reconstruct_objectshift_scenes.py \
  --base-scenes-dir /path/to/data/scenes \
  --patch-dir data/ood_object_shift/scene_patches \
  --output-dir /path/to/reconstructed/scenes
```

The script checks both the required base template and the reconstructed target
using canonical SHA-256 digests.
