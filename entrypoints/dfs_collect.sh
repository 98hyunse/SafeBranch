#!/bin/bash
# Vision-Reflection-Guided DFS 학습 데이터 수집 스크립트
#
# 사용법:
#   bash entrypoints/dfs_collect.sh                      # sequential
#   bash entrypoints/dfs_collect.sh 2                    # 2 tasks in parallel
#   bash entrypoints/dfs_collect.sh 1 task_list.txt      # custom task list
#
# 각 태스크는 results_dfs/<task_name>/ 에 저장됨:
#   report.json           — 최종 평가 메트릭
#   _trace.json           — DFS 탐색 트리 (Phase 1/2/3 이벤트)
#   golden_trajectory.json — 성공 궤적 (존재 시)
#   obs/                  — 매 스텝 관측 이미지
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "$REPO_ROOT"

if ! python -c "import omnigibson" >/dev/null 2>&1; then
    echo "ERROR: OmniGibson is not importable in the active Python environment."
    echo "       Install the upstream runtime and activate that environment first."
    exit 1
fi

export NUM_GPUS=1
export PYTHONPATH="$REPO_ROOT:${PYTHONPATH:-}"
export OMNIGIBSON_HEADLESS=1

# API key 및 환경변수 로드. run_experiment.sh가 critic profile에서 OPENAI_API_BASE/KEY를
# 이미 깔아준 상태로 호출되면 env.sh가 그 값을 OpenAI 기본값으로 덮어쓰지 않도록 가드.
if [ -f "entrypoints/env.sh" ] && [ -z "${OPENAI_API_KEY:-}" ]; then
    source entrypoints/env.sh
fi

# ── 설정 ─────────────────────────────────────────────────────────────────────
# Actor: 로컬 vLLM 서버 (기본). LOCAL_LLM_SERVE=0 이면 OpenAI close-source route
# (plan_agent.py에서 OPENAI_API_KEY/BASE 사용) — gpt-4o 등 OpenAI 모델 actor용.
MODEL_NAME_OR_PATH="${MODEL_NAME_OR_PATH:-Qwen/Qwen3-VL-32B-Instruct}"
SERVER_IP="${SERVER_IP:-http://127.0.0.1:8000/v1}"
SERVER_KEY="${SERVER_KEY:-EMPTY}"
LOCAL_LLM_SERVE="${LOCAL_LLM_SERVE:-1}"

# Critic: GPT-4o (OPENAI_API_KEY 환경변수 필요)
# 다른 critic으로 swap 시 환경변수 CRITIC_MODEL + OPENAI_API_BASE + OPENAI_API_KEY 셋만 갈아끼우면 됨.
CRITIC_MODEL="${CRITIC_MODEL:-gpt-4o}"

# DFS 파라미터
PRM_THRESHOLD=3
MAX_PHASE2_RETRIES=3
MAX_BEFORE_RETRIES=4
MAX_EXECUTION_RETRIES=6
MAX_PHASE3_RECURSION=6
# Phase3 carousel detection threshold. 0 = disable (rely solely on
# MAX_PHASE3_RECURSION). 1 = legacy behavior (trigger when same
# (trigger, culprit_step) pair seen 2x). N = trigger on (N+1)-th occurrence.
CAROUSEL_THRESHOLD=${CAROUSEL_THRESHOLD:-1}
MAX_STEPS=30
# Observation 옵션 (1: bbox on, 0: off)
DRAW_BBOX_2D="${DRAW_BBOX_2D:-1}"

ACTOR_PROMPT_VERSION="${ACTOR_PROMPT_VERSION:-v4}"
PRM_PROMPT_VERSION="${PRM_PROMPT_VERSION:-v3}"
BEFORE_BDDL_PROMPT_VERSION="${BEFORE_BDDL_PROMPT_VERSION:-v4}"
TASK_FAIL_PROMPT_VERSION="${TASK_FAIL_PROMPT_VERSION:-v0}"
TERM_SAFETY_PROMPT_VERSION="${TERM_SAFETY_PROMPT_VERSION:-v4}"
# Hybrid actor v2 fallback: -1 = never fall back, K>=0 = after K episode-wide
# rejections of the same action, skip retry → force-execute via no_terminate.
ACTOR_RETRY_BLOCK_AFTER_K=${ACTOR_RETRY_BLOCK_AFTER_K:--1}
# Base actor (planning) LLM temperature. Last 2 phase3 recursion depths add +0.3.
ACTOR_TEMPERATURE=${ACTOR_TEMPERATURE:-0.0}

OUTPUT_DIR="${OUTPUT_DIR:-./results_dfs_4o/Qwen2.5/}"
WORK_DIR="${WORK_DIR:-./work_dir}"
# Task config directory override (default: data/tasks). Set to "$REPO_ROOT/data/tasks2" for tasks2.
TASK_CONFIG_DIR="${TASK_CONFIG_DIR:-$REPO_ROOT/data/tasks}"
BDDL_DIR="${BDDL_DIR:-${OG_EGO_PRIM_BDDL_DIR:-$REPO_ROOT/data/bddl}}"
SCENE_DIR="${SCENE_DIR:-${OG_EGO_PRIM_SCENES_DIR:-$REPO_ROOT/data/scenes}}"

DATA_PARALLEL=${1:-2}
TASK_LIST=${2:-"$REPO_ROOT/entrypoints/task_list.txt"}
# Set DFS_STDOUT=0 to suppress task logs from console (file-only logging).
DFS_STDOUT=${DFS_STDOUT:-0}
# Set USE_PRM=0 to disable VisionPRM (skip Phase 1-a / Phase 2).
USE_PRM=${USE_PRM:-0}
# Set EVAL_OPEN_EXEC=1 to swallow execution errors without corrective/termination (eval_open style).
EVAL_OPEN_EXEC=${EVAL_OPEN_EXEC:-1}
# Set NO_BEFORE_BDDL=1 to skip Phase 1-b BeforeBDDLReflector check.
NO_BEFORE_BDDL=${NO_BEFORE_BDDL:-0}
# Set NO_PHASE3=1 to skip Phase 3 deep backtrack (TaskFail + TermSafety reflectors).
NO_PHASE3=${NO_PHASE3:-0}
# Set NO_TASK_FAIL_RECOVERY=1 to skip task_fail deep_backtrack (terminate on BDDL goal miss).
# term_safety 회복은 그대로 시도. critic-free 모드와 호환.
NO_TASK_FAIL_RECOVERY=${NO_TASK_FAIL_RECOVERY:-0}
# Set NO_TERMINATE_ON_RETRY_EXHAUST=1 to accept current history as golden when BeforeBDDL retry / Phase3 recursion exhausts.
NO_TERMINATE_ON_RETRY_EXHAUST=${NO_TERMINATE_ON_RETRY_EXHAUST:-1}
# Set NO_STALL=1 to disable the stall detector (5-step repeated action → stall_detected). Useful for ablation.
NO_STALL=${NO_STALL:-0}
# RIP baseline B2 'Guard'. off (default) = oracle BDDL per-step trigger.
# gpt4o = per-step GuardClassifier replaces the unsafe trigger.
GUARD_MODE=${GUARD_MODE:-off}
# Guard verifier backend (only used when GUARD_MODE != off).
#   gpt4o  = close-source GPT-4o (uses OPENAI_API_KEY/BASE + --critic_model)
#   qwen3  = local Qwen3-VL vLLM at GUARD_VERIFIER_IP (default = actor SERVER_IP)
# RIP fairness: both baselines use the SAME local Qwen3-VL verifier (qwen3).
GUARD_VERIFIER=${GUARD_VERIFIER:-gpt4o}
# vLLM endpoint for the qwen3 guard verifier. Defaults to the actor server so
# the verifier reuses the same OpenAI-compatible endpoint as the actor.
GUARD_VERIFIER_IP=${GUARD_VERIFIER_IP:-$SERVER_IP}
# RIP baseline B3 'Lookahead Search'. off (default) = standard DFS.
# lookahead = per-step depth-1 search (k candidates → 1-step rollout →
# Qwen3 SafetyValue → commit best). SEARCH_K = candidates per decision step.
SEARCH_MODE=${SEARCH_MODE:-off}
SEARCH_K=${SEARCH_K:-3}
# vLLM endpoint for the Qwen3 SafetyValue. Defaults to the actor server (RIP
# fairness: same local verifier as the Guard baseline).
SEARCH_VALUE_IP=${SEARCH_VALUE_IP:-$SERVER_IP}
# Per-task wall-clock timeout (seconds) — loose safety net only.  Primary
# protection is the per-step timeout below; this just caps the whole task in
# pathological cases (e.g. C-level hangs SIGALRM can't interrupt).  Override via env.
TASK_TIMEOUT=${TASK_TIMEOUT:-3600}
# Per-step wall-clock timeout (seconds) passed to --step_timeout_sec.  Kills
# a single executor.execute_plan call (DFS or formal_replay) that hangs,
# rolls the sim back to the pre-step snapshot, and lets DFS backtrack/retry.
# Default 1200 (20 min).  0 disables per-step timeout entirely.
STEP_TIMEOUT=${STEP_TIMEOUT:-1200}
# ─────────────────────────────────────────────────────────────────────────────

mkdir -p logs
START_TIME=$(date +%Y%m%d-%H:%M:%S)
LOG_DIR="${LOG_DIR:-logs/dfs_collect_${START_TIME}}"
mkdir -p "$LOG_DIR"
LOG_FILE="$LOG_DIR/_summary.log"

echo "════════════════════════════════════════════"
echo "  DFS Collect"
if [ "$LOCAL_LLM_SERVE" = "1" ]; then
    echo "  Actor:      $MODEL_NAME_OR_PATH (vLLM: $SERVER_IP)"
else
    echo "  Actor:      $MODEL_NAME_OR_PATH (OpenAI: ${OPENAI_API_BASE:-https://api.openai.com/v1})"
fi
echo "  Critic:     $CRITIC_MODEL"
echo "  Task list:  $TASK_LIST"
echo "  Task cfg:   $TASK_CONFIG_DIR"
echo "  BDDL dir:   $BDDL_DIR"
echo "  Scene dir:  $SCENE_DIR"
echo "  Output:     $OUTPUT_DIR"
echo "  Parallel:   $DATA_PARALLEL"
echo "  Draw bbox:  $DRAW_BBOX_2D"
echo "  PRM thr:    $PRM_THRESHOLD  N=$MAX_PHASE2_RETRIES  B=$MAX_BEFORE_RETRIES  E=$MAX_EXECUTION_RETRIES  K=$MAX_PHASE3_RECURSION"
echo "  Log dir:    $LOG_DIR"
echo "  Summary:    $LOG_FILE"
echo "════════════════════════════════════════════"

# ── Task 목록 읽기 ──────────────────────────────────────────────────────────
if [ ! -f "$TASK_LIST" ]; then
    echo "ERROR: task list not found: $TASK_LIST"
    exit 1
fi
if [ ! -d "$TASK_CONFIG_DIR" ]; then
    echo "ERROR: task config dir not found: $TASK_CONFIG_DIR"
    exit 1
fi
if [ ! -d "$BDDL_DIR" ]; then
    echo "ERROR: BDDL dir not found: $BDDL_DIR"
    exit 1
fi

export OG_EGO_PRIM_TASKS_DIR="$TASK_CONFIG_DIR"
export OG_EGO_PRIM_BDDL_DIR="$BDDL_DIR"
export OG_EGO_PRIM_SCENES_DIR="$SCENE_DIR"

mapfile -t TASKS < <(grep -v '^\s*$' "$TASK_LIST" | sed 's/\.json$//')
echo "Total tasks: ${#TASKS[@]}"

# ── 실행 함수 ────────────────────────────────────────────────────────────────
run_task() {
    local task_name="$1"
    local task_idx="$2"
    local task_total="$3"
    local task_log="$LOG_DIR/${task_name}.log"
    local BBOX_ARG=()
    if [ "$DRAW_BBOX_2D" = "1" ]; then
        BBOX_ARG+=(--draw_bbox_2d)
    fi
    local NO_PRM_ARG=()
    [ "$USE_PRM" = "0" ] && NO_PRM_ARG+=(--no_prm)
    local EVAL_OPEN_EXEC_ARG=()
    [ "$EVAL_OPEN_EXEC" = "1" ] && EVAL_OPEN_EXEC_ARG+=(--eval_open_exec)
    local NO_BEFORE_BDDL_ARG=()
    [ "$NO_BEFORE_BDDL" = "1" ] && NO_BEFORE_BDDL_ARG+=(--no_before_bddl)
    local NO_PHASE3_ARG=()
    [ "$NO_PHASE3" = "1" ] && NO_PHASE3_ARG+=(--no_phase3)
    local NO_TASK_FAIL_RECOVERY_ARG=()
    [ "$NO_TASK_FAIL_RECOVERY" = "1" ] && NO_TASK_FAIL_RECOVERY_ARG+=(--no_task_fail_recovery)
    local NO_TERMINATE_ARG=()
    [ "$NO_TERMINATE_ON_RETRY_EXHAUST" = "1" ] && NO_TERMINATE_ARG+=(--no_terminate_on_retry_exhaust)
    local NO_STALL_ARG=()
    [ "$NO_STALL" = "1" ] && NO_STALL_ARG+=(--no_stall)
    # RIP baseline B2 'Guard': GUARD_MODE=gpt4o → per-step safety guard.
    local GUARD_MODE_ARG=()
    [ -n "$GUARD_MODE" ] && GUARD_MODE_ARG+=(--guard_mode "$GUARD_MODE")
    # Guard verifier backend (gpt4o | qwen3). When qwen3, also pass the vLLM IP.
    local GUARD_VERIFIER_ARG=()
    if [ -n "$GUARD_VERIFIER" ]; then
        GUARD_VERIFIER_ARG+=(--guard_verifier "$GUARD_VERIFIER")
        [ "$GUARD_VERIFIER" = "qwen3" ] && GUARD_VERIFIER_ARG+=(--guard_verifier_ip "$GUARD_VERIFIER_IP")
    fi
    # RIP baseline B3 'Lookahead Search'.
    local SEARCH_ARG=()
    if [ "$SEARCH_MODE" != "off" ]; then
        SEARCH_ARG+=(--search_mode "$SEARCH_MODE" --search_k "$SEARCH_K" --search_value_ip "$SEARCH_VALUE_IP")
    fi
    local LOCAL_LLM_SERVE_ARG=()
    [ "$LOCAL_LLM_SERVE" = "1" ] && LOCAL_LLM_SERVE_ARG+=(--local_llm_serve)

    echo "[$(date +%H:%M:%S)] [${task_idx}/${task_total}] START  $task_name" | tee -a "$LOG_FILE"

    # Keep script alive even when one task process exits non-zero
    # (e.g., simulator shutdown crash), so the next task can continue.
    set +e
    if [ "$DFS_STDOUT" = "1" ]; then
        setsid timeout --kill-after=60 "$TASK_TIMEOUT" python -m og_ego_prim.cli.dfs_collect \
            --task_name       "$task_name" \
            --model           "$MODEL_NAME_OR_PATH" \
            "${LOCAL_LLM_SERVE_ARG[@]}" \
            --local_serve_ip  "$SERVER_IP" \
            --local_serve_key "$SERVER_KEY" \
            --critic_model    "$CRITIC_MODEL" \
            --prm_threshold   "$PRM_THRESHOLD" \
            --max_phase2_retries "$MAX_PHASE2_RETRIES" \
            --max_before_retries "$MAX_BEFORE_RETRIES" \
            --max_execution_retries "$MAX_EXECUTION_RETRIES" \
            --max_phase3_recursion "$MAX_PHASE3_RECURSION" \
            --carousel_threshold "$CAROUSEL_THRESHOLD" \
            --max_steps       "$MAX_STEPS" \
            --step_timeout_sec "$STEP_TIMEOUT" \
            --output_dir      "$OUTPUT_DIR" \
            --work_dir        "$WORK_DIR" \
            --actor_prompt_version       "$ACTOR_PROMPT_VERSION" \
            --prm_prompt_version         "$PRM_PROMPT_VERSION" \
            --before_bddl_prompt_version "$BEFORE_BDDL_PROMPT_VERSION" \
            --task_fail_prompt_version   "$TASK_FAIL_PROMPT_VERSION" \
            --term_safety_prompt_version "$TERM_SAFETY_PROMPT_VERSION" \
            --actor_retry_block_after_k  "$ACTOR_RETRY_BLOCK_AFTER_K" \
            --actor_temperature          "$ACTOR_TEMPERATURE" \
            --emit_canonical \
            "${BBOX_ARG[@]}" \
            "${NO_PRM_ARG[@]}" \
            "${EVAL_OPEN_EXEC_ARG[@]}" \
            "${NO_BEFORE_BDDL_ARG[@]}" \
            "${NO_PHASE3_ARG[@]}" \
            "${NO_TASK_FAIL_RECOVERY_ARG[@]}" \
            "${NO_TERMINATE_ARG[@]}" \
            "${NO_STALL_ARG[@]}" \
            "${GUARD_MODE_ARG[@]}" \
            "${GUARD_VERIFIER_ARG[@]}" \
            "${SEARCH_ARG[@]}" \
            2>&1 | tee -a "$task_log"
        local exit_code=${PIPESTATUS[0]}
    else
        setsid timeout --kill-after=60 "$TASK_TIMEOUT" python -m og_ego_prim.cli.dfs_collect \
            --task_name       "$task_name" \
            --model           "$MODEL_NAME_OR_PATH" \
            "${LOCAL_LLM_SERVE_ARG[@]}" \
            --local_serve_ip  "$SERVER_IP" \
            --local_serve_key "$SERVER_KEY" \
            --critic_model    "$CRITIC_MODEL" \
            --prm_threshold   "$PRM_THRESHOLD" \
            --max_phase2_retries "$MAX_PHASE2_RETRIES" \
            --max_before_retries "$MAX_BEFORE_RETRIES" \
            --max_execution_retries "$MAX_EXECUTION_RETRIES" \
            --max_phase3_recursion "$MAX_PHASE3_RECURSION" \
            --carousel_threshold "$CAROUSEL_THRESHOLD" \
            --max_steps       "$MAX_STEPS" \
            --step_timeout_sec "$STEP_TIMEOUT" \
            --output_dir      "$OUTPUT_DIR" \
            --work_dir        "$WORK_DIR" \
            --actor_prompt_version       "$ACTOR_PROMPT_VERSION" \
            --prm_prompt_version         "$PRM_PROMPT_VERSION" \
            --before_bddl_prompt_version "$BEFORE_BDDL_PROMPT_VERSION" \
            --task_fail_prompt_version   "$TASK_FAIL_PROMPT_VERSION" \
            --term_safety_prompt_version "$TERM_SAFETY_PROMPT_VERSION" \
            --actor_retry_block_after_k  "$ACTOR_RETRY_BLOCK_AFTER_K" \
            --actor_temperature          "$ACTOR_TEMPERATURE" \
            --emit_canonical \
            "${BBOX_ARG[@]}" \
            "${NO_PRM_ARG[@]}" \
            "${EVAL_OPEN_EXEC_ARG[@]}" \
            "${NO_BEFORE_BDDL_ARG[@]}" \
            "${NO_PHASE3_ARG[@]}" \
            "${NO_TASK_FAIL_RECOVERY_ARG[@]}" \
            "${NO_TERMINATE_ARG[@]}" \
            "${NO_STALL_ARG[@]}" \
            "${GUARD_MODE_ARG[@]}" \
            "${GUARD_VERIFIER_ARG[@]}" \
            "${SEARCH_ARG[@]}" \
            >> "$task_log" 2>&1
        local exit_code=$?
    fi
    set -e

    if [ $exit_code -eq 0 ]; then
        echo "[$(date +%H:%M:%S)] [${task_idx}/${task_total}] DONE   $task_name" | tee -a "$LOG_FILE"
    else
        echo "[$(date +%H:%M:%S)] [${task_idx}/${task_total}] FAILED $task_name (exit $exit_code)" | tee -a "$LOG_FILE"
    fi

    # Drop a marker file for timeout / segfault so downstream tools (pair-builder,
    # viewer index) can distinguish "SIGKILLed mid-DFS" from "proper failure".
    if [ $exit_code -ne 0 ]; then
        local marker="$OUTPUT_DIR/$task_name/_run_status.json"
        local status="failed"
        if [ $exit_code -eq 124 ]; then status="timeout"
        elif [ $exit_code -eq 139 ]; then status="segfault"
        fi
        mkdir -p "$OUTPUT_DIR/$task_name" 2>/dev/null || true
        printf '{"status": "%s", "exit_code": %d}\n' "$status" "$exit_code" > "$marker" 2>/dev/null || true
    fi
    return $exit_code
}

TASK_TOTAL=${#TASKS[@]}
export TASK_TOTAL
export -f run_task
export LOG_FILE MODEL_NAME_OR_PATH SERVER_IP SERVER_KEY CRITIC_MODEL
export PRM_THRESHOLD MAX_PHASE2_RETRIES MAX_BEFORE_RETRIES MAX_EXECUTION_RETRIES MAX_PHASE3_RECURSION CAROUSEL_THRESHOLD MAX_STEPS
export OUTPUT_DIR WORK_DIR
export ACTOR_PROMPT_VERSION PRM_PROMPT_VERSION BEFORE_BDDL_PROMPT_VERSION TASK_FAIL_PROMPT_VERSION TERM_SAFETY_PROMPT_VERSION
export ACTOR_RETRY_BLOCK_AFTER_K ACTOR_TEMPERATURE
export USE_PRM DRAW_BBOX_2D EVAL_OPEN_EXEC NO_BEFORE_BDDL NO_PHASE3 NO_TASK_FAIL_RECOVERY NO_TERMINATE_ON_RETRY_EXHAUST NO_STALL

# ── 병렬/순차 실행 ─────────────────────────────────────────────────────────
if [ "$DATA_PARALLEL" -le 1 ]; then
    # Sequential
    FAILED_COUNT=0
    TASK_IDX=0
    for task in "${TASKS[@]}"; do
        TASK_IDX=$((TASK_IDX + 1))
        if ! run_task "$task" "$TASK_IDX" "$TASK_TOTAL"; then
            FAILED_COUNT=$((FAILED_COUNT + 1))
        fi
    done
else
    # Parallel with DATA_PARALLEL concurrent jobs
    # Uses a simple semaphore via background jobs + wait
    MAX_JOBS=$DATA_PARALLEL
    TASK_IDX=0
    PIDS=()
    TASK_NAMES=()
    FAILED_COUNT=0

    for task in "${TASKS[@]}"; do
        # Wait if at capacity
        while [ "${#PIDS[@]}" -ge "$MAX_JOBS" ]; do
            NEW_PIDS=()
            NEW_TASKS=()
            for pid in "${PIDS[@]}"; do
                if kill -0 "$pid" 2>/dev/null; then
                    NEW_PIDS+=("$pid")
                    # Keep matching task name by index
                    idx=0
                    for p in "${PIDS[@]}"; do
                        if [ "$p" = "$pid" ]; then
                            NEW_TASKS+=("${TASK_NAMES[$idx]}")
                            break
                        fi
                        idx=$((idx + 1))
                    done
                fi
            done
            PIDS=("${NEW_PIDS[@]}")
            TASK_NAMES=("${NEW_TASKS[@]}")
            [ "${#PIDS[@]}" -ge "$MAX_JOBS" ] && sleep 10
        done

        TASK_IDX=$((TASK_IDX + 1))
        run_task "$task" "$TASK_IDX" "$TASK_TOTAL" &
        PIDS+=($!)
        TASK_NAMES+=("$task")
        sleep 2  # stagger launches slightly
    done

    # Wait for all remaining
    set +e
    for i in "${!PIDS[@]}"; do
        pid="${PIDS[$i]}"
        task="${TASK_NAMES[$i]}"
        wait "$pid"
        rc=$?
        if [ "$rc" -ne 0 ]; then
            FAILED_COUNT=$((FAILED_COUNT + 1))
            echo "[$(date +%H:%M:%S)] FAILED $task (exit $rc)" | tee -a "$LOG_FILE"
        fi
    done
    set -e
fi

echo ""
echo "[$(date +%H:%M:%S)] All tasks finished. Results in: $OUTPUT_DIR"
echo "Log: $LOG_FILE"
echo "Failed tasks: ${FAILED_COUNT:-0}/${#TASKS[@]}"

# Rebuild _index.json at the shell level so tasks that were SIGKILLed /
# segfaulted (and therefore never reached dfs_collect.py's internal refresh)
# still show up in viewers.  Scans every subdir with a _trace.json, so even
# partial checkpoint traces are indexed.  Standalone module — no OG import.
echo "[$(date +%H:%M:%S)] Rebuilding $OUTPUT_DIR/_index.json ..." | tee -a "$LOG_FILE"
python -m og_ego_prim.cli.refresh_index "$OUTPUT_DIR" \
    2>&1 | tee -a "$LOG_FILE" || echo "[WARN] _index.json rebuild failed" | tee -a "$LOG_FILE"

# Auto-emit canonical metrics summary so every run ends with TR/SR/SSR/IS Failure
# numbers in the log.  Skippable via SKIP_AUTO_SUMMARY=1 (e.g. for partial runs).
if [[ "${SKIP_AUTO_SUMMARY:-0}" != "1" ]]; then
    echo "" | tee -a "$LOG_FILE"
    echo "[$(date +%H:%M:%S)] Computing canonical summary ..." | tee -a "$LOG_FILE"
    mkdir -p "$OUTPUT_DIR/_report"
    # Write into _report/ (folder shows up at top in file viewers — easy to find)
    # and keep legacy top-level copy for existing tooling.
    python -m og_ego_prim.cli.summary_dfs_canonical --result_dir "$OUTPUT_DIR" \
        --output "$OUTPUT_DIR/_report/canonical_summary.json" \
        2>&1 | tee -a "$LOG_FILE" || echo "[WARN] canonical summary failed" | tee -a "$LOG_FILE"
    cp "$OUTPUT_DIR/_report/canonical_summary.json" "$OUTPUT_DIR/report_all_canonical.json" 2>/dev/null || true
    echo "[$(date +%H:%M:%S)] Canonical summary → $OUTPUT_DIR/_report/canonical_summary.json (+ legacy report_all_canonical.json)" | tee -a "$LOG_FILE"
fi

# Auto-build standalone HTML report ($OUTPUT_DIR/_report/index.html). Cheap
# (~1s for 161 tasks, ~5MB).  Skippable via SKIP_AUTO_HTML=1.
if [[ "${SKIP_AUTO_HTML:-0}" != "1" ]]; then
    echo "" | tee -a "$LOG_FILE"
    echo "[$(date +%H:%M:%S)] Building HTML report ..." | tee -a "$LOG_FILE"
    python -m og_ego_prim.cli.dfs_html_report --results_dir "$OUTPUT_DIR" --offline --quiet \
        2>&1 | tee -a "$LOG_FILE" || echo "[WARN] HTML report build failed" | tee -a "$LOG_FILE"
    echo "[$(date +%H:%M:%S)] HTML report → $OUTPUT_DIR/_report/index.html" | tee -a "$LOG_FILE"
fi
