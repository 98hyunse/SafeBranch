# OOD Benchmarks

This directory contains the two out-of-distribution evaluation sets used by
SafeBranch. Neither package includes OmniGibson or BEHAVIOR 3D assets.

- `ood_object_shift`: 147 environment variants created by adding a neutral or
  hazard-associated distractor object. Scene changes are distributed as small
  patches against the corresponding upstream scene templates.
- `ood_task_shift`: 159 target-object-substitution tasks. The paper reports the
  common successfully completed subset listed in `paper_eval_138.txt`.

The required upstream simulator assets must be obtained under their own terms.
