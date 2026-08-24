import os

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))

WORK_DIR = os.environ.get("OG_EGO_PRIM_WORK_DIR", os.path.join(REPO_ROOT, 'work_dir'))

DATA = os.environ.get("OG_EGO_PRIM_DATA_DIR", os.path.join(REPO_ROOT, 'data'))
METAS = os.environ.get("OG_EGO_PRIM_METAS_DIR", os.path.join(DATA, 'metas'))
TASKS = os.environ.get("OG_EGO_PRIM_TASKS_DIR", os.path.join(DATA, 'tasks'))
CAMERAS = os.environ.get("OG_EGO_PRIM_CAMERAS_DIR", os.path.join(DATA, 'cameras'))
BDDLS = os.environ.get("OG_EGO_PRIM_BDDL_DIR", os.path.join(DATA, 'bddl'))
SCENES = os.environ.get("OG_EGO_PRIM_SCENES_DIR", os.path.join(DATA, 'scenes'))
