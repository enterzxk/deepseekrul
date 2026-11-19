#!/usr/bin/env bash
# =============================================================================
# Phase 1c queue: RHSE-TSH RUL recovery grid + clean EOL false-alarm rate.
#
# Frozen grid (do not edit without a new freeze id):
#   subsets FD001 FD003 | faults bias drift stuck noise | mag 2 3 5
#   seeds 101 202 303 | sensors 0,5,10 | gamma 1.30 | onset 0.5
#   2 * 4 * 3 * 3 = 72 RUL jobs
#   + tsh_eol_fpr on FD001/FD003 x seeds 101/202/303 = 6 jobs
#   = 78 jobs, executed strictly one at a time on cuda:0.
#
# Data policy: train/validation only. No official test file is read, named or
# evaluated by this queue; the test split stays sealed.
#
# Commands:
#   start    detach under nohup and return immediately
#   run      run in the foreground (one job at a time)
#   status   complete / failed / started-not-finished / not-started, current job
#   collect  key metric lines of every finished job
#   tsv      same data as tab-separated rows
#   mean     3-seed mean/min/max recovery per (subset, fault, mag, arm)
#   stop     graceful stop: finish the current job, then exit
#
# Idempotency: a job whose log already contains a complete summary is skipped.
#   rul: clean R0 + R3/R5/R6_hybrid/R7_filter recovery lines
#   eol: the EOL_cycle_flag_rate line
# Incomplete logs are appended to, never truncated, so failures stay on disk
# and a re-run retries exactly the unfinished jobs.
# =============================================================================
set -Eeuo pipefail

SCRIPT_PATH="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)/$(basename -- "${BASH_SOURCE[0]}")"

SPRINT24="${SPRINT24:-${HOME}/sprint24}"
PY="${PYTHON:-/home/user/miniconda3/envs/ncmapss/bin/python}"
LOG_ROOT="${PHASE1C_LOG_ROOT:-${SPRINT24}/experiments/phase1c}"
LOG_DIR="${LOG_ROOT}/logs"
STATE_DIR="${LOG_ROOT}/state"
SUMMARY_DIR="${LOG_ROOT}/summary"
LOCK_DIR="${STATE_DIR}/.queue.lock"
STOP_FILE="${STATE_DIR}/STOP"
CURRENT_FILE="${STATE_DIR}/current_job"
CURRENT_START="${STATE_DIR}/current_start"
QUEUE_LOG="${LOG_ROOT}/queue.log"

export PYTHONPATH="${SPRINT24}${PYTHONPATH:+:${PYTHONPATH}}"
export PYTHONUNBUFFERED=1
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

# Frozen grid.
SUBSETS=(FD001 FD003)
FAULTS=(bias drift stuck noise)
MAGS=(2 3 5)
SEEDS=(101 202 303)
SENSORS="0,5,10"
GAMMA="1.30"
ONSET="0.5"

usage() {
  cat <<'EOF'
usage: bash scripts/queue_phase1c.sh <command>

  start    detach the queue under nohup and return immediately
  run      run the queue in the foreground (one job at a time)
  status   progress: complete / failed / started / not started, current job
  collect  print the key metric lines of every finished job
  tsv      same data as tab-separated rows
  mean     3-seed mean/min/max recovery per (subset, fault, mag, arm)
  stop     graceful stop: finish the current job, then exit
  help     this message
EOF
}

ensure_layout() {
  case "$LOG_ROOT" in
    */experiments/phase1c) ;;
    *)
      printf 'refusing LOG_ROOT=%s (must end in /experiments/phase1c)\n' "$LOG_ROOT" >&2
      exit 2
      ;;
  esac
  mkdir -p "$LOG_DIR" "$STATE_DIR" "$SUMMARY_DIR"
}

root_for() {
  case "$1" in
    FD001) printf '%s\n' '/home/user/sprint24/experiments/hc/_gate' ;;
    FD003) printf '%s\n' '/home/user/sprint24/experiments/hc/d80' ;;
    *)
      printf 'unknown subset: %s\n' "$1" >&2
      return 1
      ;;
  esac
}

# One line per job: kind <TAB> subset <TAB> fault <TAB> mag <TAB> seed.
# EOL jobs come first: they finish in seconds and validate the interpreter,
# the roots and gamma=1.30 before the 72 long RUL jobs start.
emit_jobs() {
  local s f m seed
  # EOL FPR is seed-independent (it does not use the trained net); one job per subset.
  for s in "${SUBSETS[@]}"; do
    printf 'eol\t%s\t-\t-\t101\n' "$s"
  done
  for s in "${SUBSETS[@]}"; do
    for f in "${FAULTS[@]}"; do
      for m in "${MAGS[@]}"; do
        # stuck is a z-space constant and ignores mag; only one cell is unique.
        if [[ "$f" == 'stuck' && "$m" != '3' ]]; then
          continue
        fi
        for seed in "${SEEDS[@]}"; do
          printf 'rul\t%s\t%s\t%s\t%s\n' "$s" "$f" "$m" "$seed"
        done
      done
    done
  done
}

job_id_for() { # kind subset fault mag seed
  if [[ "$1" == 'eol' ]]; then
    printf 'eol_%s_s%s\n' "$2" "$5"
  else
    printf 'rul_%s_%s_m%s_s%s\n' "$2" "$3" "$4" "$5"
  fi
}

log_for() { # kind subset fault mag seed
  printf '%s/%s.log\n' "$LOG_DIR" "$(job_id_for "$1" "$2" "$3" "$4" "$5")"
}

# A complete summary, not merely a non-empty log.
is_complete() {
  local kind="$1" log="$2"
  [[ -s "$log" ]] || return 1
  case "$kind" in
    rul)
      grep -q '^clean R0 ' "$log" || return 1
      grep -q '^R3 recovery ' "$log" || return 1
      grep -q '^R5 recovery ' "$log" || return 1
      grep -q '^R6_hybrid recovery ' "$log" || return 1
      grep -q '^R7_filter recovery ' "$log" || return 1
      ;;
    eol)
      grep -q 'EOL_cycle_flag_rate=' "$log" || return 1
      ;;
    *)
      return 1
      ;;
  esac
  return 0
}

# Print only the last "=== RUN" block, so appended retries never duplicate.
last_block() {
  awk '
    /^=== RUN / { n = 0; next }
    { buf[++n] = $0 }
    END { for (i = 1; i <= n; i++) print buf[i] }
  ' "$1"
}

lock_pid() {
  cat "${LOCK_DIR}/pid" 2>/dev/null || true
}

queue_running() {
  local pid
  pid="$(lock_pid)"
  [[ -n "$pid" ]] && kill -0 "$pid" 2>/dev/null
}

lock_started() {
  cat "${LOCK_DIR}/started" 2>/dev/null || printf '?'
}

job_elapsed() {
  local start
  start="$(cat "$CURRENT_START" 2>/dev/null || true)"
  if [[ -z "$start" ]]; then
    printf '?'
    return 0
  fi
  printf '%s' "$(( $(date +%s) - start ))"
}

acquire_lock() {
  local other
  if ! mkdir "$LOCK_DIR" 2>/dev/null; then
    other="$(lock_pid)"
    if [[ -n "$other" ]] && kill -0 "$other" 2>/dev/null; then
      printf 'queue already running (pid %s); one job at a time -- refusing a second instance\n' \
        "$other" >&2
      exit 3
    fi
    printf 'clearing stale lock (pid %s is not alive)\n' "${other:-unknown}" >&2
    rm -f "${LOCK_DIR}/pid" 2>/dev/null || true
    rmdir "$LOCK_DIR" 2>/dev/null || true
    mkdir "$LOCK_DIR" || {
      printf 'cannot create lock: %s\n' "$LOCK_DIR" >&2
      exit 3
    }
  fi
  printf '%s\n' "$$" > "${LOCK_DIR}/pid"
  date -u +%Y-%m-%dT%H:%M:%SZ > "${LOCK_DIR}/started"
  trap release_lock EXIT
}

release_lock() {
  local owner
  owner="$(lock_pid)"
  if [[ "$owner" == "$$" ]]; then
    rm -f "${LOCK_DIR}/pid" 2>/dev/null || true
    rm -f "${LOCK_DIR}/started" 2>/dev/null || true
    rm -f "$CURRENT_FILE" "$CURRENT_START" 2>/dev/null || true
    rmdir "$LOCK_DIR" 2>/dev/null || true
  fi
}

preflight() {
  local s seed ckpt missing=0

  if [[ "${SKIP_PREFLIGHT:-0}" == '1' ]]; then
    printf 'preflight: skipped (SKIP_PREFLIGHT=1)\n'
    return 0
  fi
  if [[ ! -d "$SPRINT24" ]]; then
    printf 'preflight failed: repo root not found: %s\n' "$SPRINT24" >&2
    return 1
  fi
  if [[ ! -x "$PY" ]]; then
    printf 'preflight failed: python not executable: %s\n' "$PY" >&2
    return 1
  fi
  for s in "${SUBSETS[@]}"; do
    for seed in "${SEEDS[@]}"; do
      ckpt="$(root_for "$s")/HC-D-B0-${s}-s${seed}-D1/ckpt/best.pt"
      if [[ ! -f "$ckpt" ]]; then
        printf 'preflight MISSING checkpoint: %s\n' "$ckpt" >&2
        missing=1
      fi
    done
  done
  if [[ $missing -ne 0 ]]; then
    printf 'preflight failed: seed checkpoints missing; nothing was launched.\n' >&2
    printf 'fix the roots or set SKIP_PREFLIGHT=1 to override.\n' >&2
    return 1
  fi
  return 0
}

run_job() { # kind subset fault mag seed
  local kind="$1" subset="$2" fault="$3" mag="$4" seed="$5"
  local id log root rc=0 start elapsed
  id="$(job_id_for "$kind" "$subset" "$fault" "$mag" "$seed")"
  log="$(log_for "$kind" "$subset" "$fault" "$mag" "$seed")"
  root="$(root_for "$subset")"

  local -a cmd
  if [[ "$kind" == 'rul' ]]; then
    cmd=("$PY" -u -m paper_b.modules.tsh_rul_eval
         --subset "$subset" --seed "$seed" --root "$root"
         --fault "$fault" --mag "$mag" --sensors "$SENSORS"
         --gamma "$GAMMA" --onset "$ONSET")
  else
    cmd=("$PY" -u -m paper_b.modules.tsh_eol_fpr
         --subset "$subset" --seed "$seed" --root "$root"
         --gamma "$GAMMA")
  fi
  if [[ -n "${CMAPSS_DATA_DIR:-}" ]]; then
    cmd+=(--data-dir "$CMAPSS_DATA_DIR")
  fi

  printf '%s\n' "$id" > "$CURRENT_FILE"
  date +%s > "$CURRENT_START"
  printf '=== RUN %s %s ===\n' "$id" "$(date -u +%Y-%m-%dT%H:%M:%SZ)" >> "$log"
  start=$SECONDS
  if ( cd "$SPRINT24" && "${cmd[@]}" ) >> "$log" 2>&1; then
    rc=0
  else
    rc=$?
  fi
  elapsed=$((SECONDS - start))

  if [[ $rc -eq 0 ]] && is_complete "$kind" "$log"; then
    printf 'DONE %-36s %4ds  %s\n' "$id" "$elapsed" "$log"
    return 0
  fi
  printf 'FAIL %-36s rc=%d %4ds  %s\n' "$id" "$rc" "$elapsed" "$log"
  printf '     cmd: %s\n' "${cmd[*]}"
  return 1
}

cmd_run() {
  ensure_layout
  acquire_lock
  preflight

  local total n_eol n_rul skip=0 done_n=0 fail_n=0 stopped=0
  local kind subset fault mag seed id log note=''
  total="$(emit_jobs | wc -l | tr -d ' ')"
  n_eol=$(( ${#SUBSETS[@]} * ${#SEEDS[@]} ))
  n_rul=$(( total - n_eol ))

  printf 'phase1c: %d jobs (rul %d, eol %d), one at a time, cuda:0\n' "$total" "$n_rul" "$n_eol"
  printf 'repo    : %s\n' "$SPRINT24"
  printf 'python  : %s\n' "$PY"
  printf 'logs    : %s\n' "$LOG_ROOT"

  while IFS=$'\t' read -r kind subset fault mag seed; do
    [[ -n "${kind:-}" ]] || continue

    if [[ -f "$STOP_FILE" ]]; then
      rm -f "$STOP_FILE"
      printf 'stop requested: exiting before the next job\n'
      stopped=1
      break
    fi

    id="$(job_id_for "$kind" "$subset" "$fault" "$mag" "$seed")"
    log="$(log_for "$kind" "$subset" "$fault" "$mag" "$seed")"
    if is_complete "$kind" "$log"; then
      skip=$((skip + 1))
      continue
    fi

    if run_job "$kind" "$subset" "$fault" "$mag" "$seed"; then
      done_n=$((done_n + 1))
    else
      fail_n=$((fail_n + 1))
    fi
  done < <(emit_jobs)

  rm -f "$CURRENT_FILE" "$CURRENT_START" 2>/dev/null || true
  if [[ $stopped -eq 1 ]]; then
    note=', stopped early'
  fi
  printf 'queue summary: %d ran ok, %d already complete (skipped), %d failed, %d total%s\n' \
    "$done_n" "$skip" "$fail_n" "$total" "$note"
  if [[ $fail_n -gt 0 ]]; then
    printf 're-run to retry failed/incomplete jobs; complete logs are skipped.\n'
    return 1
  fi
  return 0
}

cmd_start() {
  ensure_layout
  if queue_running; then
    printf 'queue already running (pid %s)\n' "$(lock_pid)" >&2
    return 3
  fi
  nohup bash "$SCRIPT_PATH" run >> "$QUEUE_LOG" 2>&1 &
  printf 'started queue pid %s\n' "$!"
  printf 'driver log: %s\n' "$QUEUE_LOG"
  printf 'status    : bash %s status\n' "$SCRIPT_PATH"
}

cmd_stop() {
  ensure_layout
  if ! queue_running; then
    printf 'queue is not running; nothing to stop\n'
    return 0
  fi
  : > "$STOP_FILE"
  printf 'stop requested for pid %s; it will exit after the current job\n' "$(lock_pid)"
  printf 'immediate stop: kill %s (leaves the current log incomplete; it re-runs next launch)\n' \
    "$(lock_pid)"
}

cmd_status() {
  ensure_layout

  local total=0 complete=0 failed=0 started=0 remaining=0
  local rul_total=0 rul_done=0 eol_total=0 eol_done=0
  local kind subset fault mag seed log

  while IFS=$'\t' read -r kind subset fault mag seed; do
    [[ -n "${kind:-}" ]] || continue
    total=$((total + 1))
    if [[ "$kind" == 'rul' ]]; then
      rul_total=$((rul_total + 1))
    else
      eol_total=$((eol_total + 1))
    fi
    log="$(log_for "$kind" "$subset" "$fault" "$mag" "$seed")"
    if is_complete "$kind" "$log"; then
      complete=$((complete + 1))
      if [[ "$kind" == 'rul' ]]; then
        rul_done=$((rul_done + 1))
      else
        eol_done=$((eol_done + 1))
      fi
    elif [[ -s "$log" ]] && grep -q '^Traceback' "$log"; then
      failed=$((failed + 1))
    elif [[ -s "$log" ]]; then
      started=$((started + 1))
    else
      remaining=$((remaining + 1))
    fi
  done < <(emit_jobs)

  printf 'phase1c: %d/%d complete | %d failed | %d started-not-finished | %d not started\n' \
    "$complete" "$total" "$failed" "$started" "$remaining"
  printf '  rul %d/%d   eol %d/%d\n' "$rul_done" "$rul_total" "$eol_done" "$eol_total"

  if queue_running; then
    printf 'queue: RUNNING pid %s since %s\n' "$(lock_pid)" "$(lock_started)"
    if [[ -f "$CURRENT_FILE" ]]; then
      local cur
      cur="$(cat "$CURRENT_FILE")"
      printf 'current: %s (elapsed %ss)\n' "$cur" "$(job_elapsed)"
      if [[ -f "$LOG_DIR/${cur}.log" ]]; then
        printf '  last : %s\n' "$(tail -n 1 "$LOG_DIR/${cur}.log" | cut -c1-160)"
      fi
    fi
  else
    printf 'queue: not running\n'
  fi
}

cmd_collect() {
  ensure_layout

  local kind subset fault mag seed id log
  while IFS=$'\t' read -r kind subset fault mag seed; do
    [[ -n "${kind:-}" ]] || continue
    id="$(job_id_for "$kind" "$subset" "$fault" "$mag" "$seed")"
    log="$(log_for "$kind" "$subset" "$fault" "$mag" "$seed")"
    if ! is_complete "$kind" "$log"; then
      printf '## %s  INCOMPLETE\n' "$id"
      continue
    fi
    printf '## %s\n' "$id"
    last_block "$log" \
      | grep -E '^(clean R0 |R3 recovery |R5 recovery |R6_hybrid recovery |R7_filter recovery )|EOL_any_FPR=' \
      || true
  done < <(emit_jobs)
}

cmd_tsv() {
  ensure_layout

  printf 'kind\tsubset\tfault\tmag\tseed\tmetric\tvalue\n'
  local kind subset fault mag seed log
  while IFS=$'\t' read -r kind subset fault mag seed; do
    [[ -n "${kind:-}" ]] || continue
    log="$(log_for "$kind" "$subset" "$fault" "$mag" "$seed")"
    is_complete "$kind" "$log" || continue
    if [[ "$kind" == 'rul' ]]; then
      last_block "$log" | awk -v s="$subset" -v f="$fault" -v m="$mag" -v sd="$seed" '
        $1 == "clean" && $2 == "R0" {
          v = $7; sub(/^\+/, "", v)
          printf "rul\t%s\t%s\t%s\t%s\tclean_cost\t%s\n", s, f, m, sd, v
        }
        $2 == "recovery" {
          printf "rul\t%s\t%s\t%s\t%s\t%s_recovery\t%s\n", s, f, m, sd, $1, $3
        }
      '
    else
      last_block "$log" | awk -v s="$subset" -v sd="$seed" '
        /EOL_any_FPR=/ {
          any = ""; cyc = ""
          for (i = 1; i <= NF; i++) {
            if ($i ~ /^EOL_any_FPR=/)         { split($i, a, "="); any = a[2] }
            if ($i ~ /^EOL_cycle_flag_rate=/) { split($i, b, "="); cyc = b[2] }
          }
          printf "eol\t%s\t-\t-\t%s\tEOL_any_FPR\t%s\n", s, sd, any
          printf "eol\t%s\t-\t-\t%s\tEOL_cycle_flag_rate\t%s\n", s, sd, cyc
        }
      '
    fi
  done < <(emit_jobs)
}

cmd_mean() {
  cmd_tsv | awk -F'\t' '
    NR == 1 { next }
    $6 ~ /_recovery$/ {
      k = $2 "|" $3 "|" $4 "|" $6
      sum[k] += $7; n[k]++
      v = $7 + 0
      if (!(k in lo) || v < lo[k]) lo[k] = v
      if (!(k in hi) || v > hi[k]) hi[k] = v
    }
    END {
      for (k in sum) {
        split(k, p, "|")
        printf "%s\t%s\tmag=%s\t%s\tn=%d\tmean=%.4f\tmin=%.4f\tmax=%.4f\n",
               p[1], p[2], p[3], p[4], n[k], sum[k] / n[k], lo[k], hi[k]
      }
    }
  ' | sort
}

main() {
  local cmd="${1:-run}"
  case "$cmd" in
    start)   cmd_start ;;
    run)     cmd_run ;;
    status)  cmd_status ;;
    collect) cmd_collect ;;
    tsv)     cmd_tsv ;;
    mean)    cmd_mean ;;
    stop)    cmd_stop ;;
    help|-h|--help) usage ;;
    *)
      usage >&2
      exit 64
      ;;
  esac
}

main "$@"
