set -uo pipefail

PYTHON="${PYTHON:-python}"
RESULTS_DIR="results"
LOGS_DIR="logs"

PKPD_SEEDS=25
QUAD_SEEDS=25
LORENZ_SEEDS=25

EPOCHS=""
BENCHMARKS="pkpd quadrotor lorenz"
FORCE=0
DRY_RUN=0
SMOKE=0

script_for() {
  case "$1" in
    pkpd)      echo "PKPDNN.py" ;;
    quadrotor) echo "quadrotorNN.py" ;;
    lorenz)    echo "lorenzNN.py" ;;
    *)         echo "" ;;
  esac
}

prefix_for() {
  case "$1" in
    pkpd)      echo "pkpd" ;;
    quadrotor) echo "quadrotor" ;;
    lorenz)    echo "lorenz" ;;
    *)         echo "" ;;
  esac
}

seeds_for() {
  case "$1" in
    pkpd)      echo "$PKPD_SEEDS" ;;
    quadrotor) echo "$QUAD_SEEDS" ;;
    lorenz)    echo "$LORENZ_SEEDS" ;;
    *)         echo "0" ;;
  esac
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --smoke)         SMOKE=1; EPOCHS=200; shift ;;
    --epochs)        EPOCHS="$2"; shift 2 ;;
    --benchmarks)    BENCHMARKS="$2"; shift 2 ;;
    --pkpd-seeds)    PKPD_SEEDS="$2"; shift 2 ;;
    --quad-seeds)    QUAD_SEEDS="$2"; shift 2 ;;
    --lorenz-seeds)  LORENZ_SEEDS="$2"; shift 2 ;;
    --force)         FORCE=1; shift ;;
    --dry-run)       DRY_RUN=1; shift ;;
    -h|--help)       sed -n '2,20p' "$0"; exit 0 ;;
    *) echo "Unknown argument: $1"; exit 1 ;;
  esac
done

BENCHMARKS="${BENCHMARKS//,/ }"

mkdir -p "$RESULTS_DIR" "$LOGS_DIR"

for b in $BENCHMARKS; do
  s=$(script_for "$b")
  if [[ -z "$s" ]]; then
    echo "ERROR: unknown benchmark '$b' (expected: pkpd, quadrotor, lorenz)"; exit 1
  fi
  if [[ ! -f "$s" ]]; then
    echo "ERROR: $s not found in $(pwd)"; exit 1
  fi
done

if ! command -v "$PYTHON" >/dev/null 2>&1; then
  echo "ERROR: '$PYTHON' not on PATH. Set PYTHON=... to override."; exit 1
fi

SWEEP_ID="$(date +%Y%m%d_%H%M%S)"
MANIFEST="$LOGS_DIR/sweep_${SWEEP_ID}.manifest"

{
  echo "sweep_id      : $SWEEP_ID"
  echo "started       : $(date)"
  echo "host          : $(hostname)"
  echo "cwd           : $(pwd)"
  echo "python        : $($PYTHON --version 2>&1)"
  echo "git_commit    : $(git rev-parse --short HEAD 2>/dev/null || echo 'not a git repo')"
  echo "benchmarks    : $BENCHMARKS"
  echo "epochs        : ${EPOCHS:-default}"
  echo "pkpd_seeds    : $PKPD_SEEDS"
  echo "quad_seeds    : $QUAD_SEEDS"
  echo "lorenz_seeds  : $LORENZ_SEEDS"
  echo "force         : $FORCE"
} | tee "$MANIFEST"
echo


JOBS=()
SKIPPED=0

if [[ $SMOKE -eq 1 ]]; then
  for b in $BENCHMARKS; do JOBS+=("$b 99"); done
else
  for b in $BENCHMARKS; do
    n=$(seeds_for "$b")
    p=$(prefix_for "$b")
    for ((s = 1; s <= n; s++)); do
      out="$RESULTS_DIR/${p}_seed${s}.npz"
      if [[ -f "$out" && $FORCE -eq 0 ]]; then
        echo "skip  $b seed $s  (found $out)"
        SKIPPED=$((SKIPPED + 1))
      else
        JOBS+=("$b $s")
      fi
    done
  done
fi

TOTAL=${#JOBS[@]}
echo
echo "Planned: $TOTAL run(s), $SKIPPED already complete."
[[ $TOTAL -eq 0 ]] && { echo "Nothing to do."; exit 0; }

if [[ $DRY_RUN -eq 1 ]]; then
  echo "--- dry run, job list ---"
  for j in "${JOBS[@]}"; do echo "  $j"; done
  exit 0
fi


INTERRUPTED=0
on_interrupt() {
  INTERRUPTED=1
  echo
  echo "!! Interrupted. Completed runs are in $RESULTS_DIR/;"
  echo "   re-run the same command to resume."
  exit 130
}
trap on_interrupt INT TERM

SWEEP_START=$(date +%s)
IDX=0
FAILED=()

for job in "${JOBS[@]}"; do
  read -r b s <<<"$job"
  IDX=$((IDX + 1))
  script=$(script_for "$b")
  prefix=$(prefix_for "$b")
  log="$LOGS_DIR/${prefix}_seed${s}.log"

  cmd=("$PYTHON" "$script" --seed "$s" --no-plots --outdir "$RESULTS_DIR")
  [[ -n "$EPOCHS" ]] && cmd+=(--epochs "$EPOCHS")

  echo "======================================================================"
  echo "[$IDX/$TOTAL] $b  seed $s   ($(date +%H:%M:%S))"
  echo "  ${cmd[*]}"
  echo "  log -> $log"
  echo "======================================================================"

  t0=$(date +%s)
  "${cmd[@]}" > >(tee "$log") 2>&1
  status=$?
  t1=$(date +%s)
  elapsed=$((t1 - t0))

  if [[ $status -ne 0 ]]; then
    echo "  ** FAILED (exit $status) after ${elapsed}s — see $log"
    FAILED+=("$b seed $s (exit $status)")
  else
    echo "  done in ${elapsed}s ($((elapsed / 60))m)"
  fi

  now=$(date +%s)
  avg=$(( (now - SWEEP_START) / IDX ))
  remaining=$(( avg * (TOTAL - IDX) ))
  echo "  elapsed $(( (now - SWEEP_START) / 60 ))m | est. remaining $(( remaining / 60 ))m"
  echo
done

SWEEP_END=$(date +%s)

echo "======================================================================"
echo "SWEEP COMPLETE"
echo "  runs      : $TOTAL"
echo "  skipped   : $SKIPPED"
echo "  failures  : ${#FAILED[@]}"
echo "  wall-clock: $(( (SWEEP_END - SWEEP_START) / 60 )) min"
echo "======================================================================"

if [[ ${#FAILED[@]} -gt 0 ]]; then
  echo "Failed runs:"
  for f in "${FAILED[@]}"; do echo "  - $f"; done
  echo
fi

{
  echo "finished      : $(date)"
  echo "runs          : $TOTAL"
  echo "failures      : ${#FAILED[@]}"
  echo "wall_clock_min: $(( (SWEEP_END - SWEEP_START) / 60 ))"
} >> "$MANIFEST"

"$PYTHON" - "$RESULTS_DIR" <<'PYEOF'
import glob, os, sys
import numpy as np

results_dir = sys.argv[1]
for prefix, title in [("pkpd", "PK/PD"), ("quadrotor", "Quadrotor"), ("lorenz", "Lorenz-96")]:
    files = sorted(glob.glob(os.path.join(results_dir, f"{prefix}_seed*.npz")))
    if not files:
        continue
    print(f"\n{title}  ({len(files)} seed(s))")
    print(f"  {'method':<8}{'J_true mean':>14}{'std':>12}{'diverged':>10}")
    rows = {}
    for f in files:
        d = np.load(f, allow_pickle=False)
        for m in [str(x) for x in d["methods"]]:
            key = f"{m}__J_true"
            if key in d.files:
                rows.setdefault(m, {"J": [], "div": 0})
                v = float(d[key])
                if np.isfinite(v):
                    rows[m]["J"].append(v)
                else:
                    rows[m]["div"] += 1
    for m, r in rows.items():
        if r["J"]:
            print(f"  {m:<8}{np.mean(r['J']):>14.6f}{np.std(r['J'], ddof=1) if len(r['J'])>1 else float('nan'):>12.6f}{r['div']:>10}")
        else:
            print(f"  {m:<8}{'all failed':>14}{'':>12}{r['div']:>10}")
print()
PYEOF

exit 0
