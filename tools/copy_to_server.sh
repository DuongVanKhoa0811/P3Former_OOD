#!/usr/bin/env bash
# Copy what git does not track (datasets, info pkls, checkpoints, result logs) from this machine
# to the clone of this repo on another server, over SSH. The code itself comes from GitHub.
#
# Usage, on the machine that has the data:
#   bash tools/copy_to_server.sh --host USER@HOST [--jump USER@JUMP] --repo DIR [--store DIR]
#                                [--plan | --dry-run] [--force] [TIER...] [--path P...]
#   --host     the server (an ~/.ssh/config alias works too)
#   --jump     an SSH jump host on the way, if the server is not reachable directly (ssh -J)
#   --repo     the git clone on the server
#   --store    where big data goes on the server, symlinked into the clone (as /mnt/ssd and
#              /mnt/sandisk are on server A); pick its largest disk. Default: DIR/../store
#   --plan     list what would be copied and its size, without connecting
#   --dry-run  rsync dry run: connects and compares, changes nothing
#   --force    skip the free-space check, e.g. when part of the data is already there
#   --path P   also copy P, a file or directory relative to the repo root, to the same place in
#              the clone (repeatable), e.g. a new run in work_dirs/
# Tiers (default core, or none when --path is given):
#   core           DSO dataset (128 GB), DSO info pkls, the trained DSO checkpoint (0.9 GB), small
#                  result logs (OOD baselines, grouping sweep + robust.tsv, hierarchy ablation),
#                  papers/, AGENTS.md, .superpowers/, and a pip freeze of the p3former env here
#   semantickitti  SemanticKITTI with its pkls (90 GB), checkpoint/ (the official checkpoint), and
#                  the 2xb1 SemanticKITTI runs (1.1 GB)
#   analysis       grouping's logit dumps and feature samples (~75 GB)
# Rerunning is safe: files already there are skipped, --partial resumes a half-copied file,
# files that are newer on the server are never overwritten, and nothing is deleted there.
#
# Before the first copy to a new server:
#   1. Clone the repo there with the server's own GitHub login (e.g. `gh auth login`). Never copy
#      this machine's .git/config, which can hold credentials.
#   2. Let this machine log in to every hop without a password: add its ~/.ssh/id_*.pub to
#      ~/.ssh/authorized_keys on the server and on the jump host (an administrator account on a
#      Windows OpenSSH jump host reads C:\ProgramData\ssh\administrators_authorized_keys). Log
#      in once by hand to accept the host keys: ssh -J USER@JUMP USER@HOST true
#   3. Run --plan, then --dry-run, then the copy. A full copy takes hours (the core tier took
#      13.5 h in October 2026, at ~2.8 MB/s through a Tailscale relay), so start it detached:
#      nohup setsid bash -c "bash tools/copy_to_server.sh ARGS > trash/copy_to_server.log 2>&1" \
#        < /dev/null > /dev/null 2>&1 &
set -euo pipefail
# The shell profile on server A puts a conda env's lib/ on LD_LIBRARY_PATH; the system ssh then
# loads conda's OpenSSL and refuses to start ("OpenSSL version mismatch"). Nothing here needs it.
unset LD_LIBRARY_PATH

die() { echo "$*" >&2; exit 1; }
usage() { awk 'NR > 1 && !/^#/ {exit} NR > 1 {sub(/^# ?/, ""); print}' "$0"; }
need_value() { [ "$2" -ge 2 ] || die "$1 needs a value (see --help)"; }

HOST= JUMP= REPO= STORE= MODE=copy FORCE=0 TIERS=() PATHS=()
while [ $# -gt 0 ]; do
  case $1 in
    --host) need_value "$1" $#; HOST=$2; shift ;;
    --jump) need_value "$1" $#; JUMP=$2; shift ;;
    --repo) need_value "$1" $#; REPO=${2%/}; shift ;;
    --store) need_value "$1" $#; STORE=${2%/}; shift ;;
    --path) need_value "$1" $#; PATHS+=("$2"); shift ;;
    --plan) MODE=plan ;;
    --dry-run) MODE=dry ;;
    --force) FORCE=1 ;;
    core|semantickitti|analysis) TIERS+=("$1") ;;
    -h|--help) usage; exit 0 ;;
    *) die "unknown argument: $1 (see --help)" ;;
  esac
  shift
done
[ -n "$HOST" ] && [ -n "$REPO" ] || die "--host and --repo are required (see --help)"
STORE=${STORE:-$(dirname "$REPO")/store}
[ ${#TIERS[@]} -gt 0 ] || [ ${#PATHS[@]} -gt 0 ] || TIERS=(core)
want() { local t; for t in "${TIERS[@]}"; do [ "$t" = "$1" ] && return 0; done; return 1; }

# ---- this machine (source) ----
SRC_REPO=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
SRC_DSO=$(readlink -m "$SRC_REPO/data/dso/annotations")
SRC_SKITTI=$(readlink -m "$SRC_REPO/data/semantickitti")
SRC_PY=${P3FORMER_PY:-$HOME/miniconda3/envs/p3former/bin/python}  # for the pip freeze
FREEZE=trash/pip_freeze_p3former_$(hostname -s).txt
DUMP=work_dirs/p3former_2xb1_3x_dso_ood_dump
BIG_DUMPS=(logits logits_test features_cetran features_test)  # analysis tier only
CORE_RUNS=(p3former_2xb1_3x_dso p3former_2xb1_3x_dso_ood p3former_2xb1_3x_dso_ood_dump_test
           p3former_2xb1_3x_dso_ood_hier)
SKITTI_RUNS=(p3former_2xb1_3x_semantickitti p3former_2xb1_3x_semantickitti_ood)
NOTES=()  # untracked notes, copied if present
for f in papers AGENTS.md .superpowers; do
  if [ -e "$SRC_REPO/$f" ]; then NOTES+=("$SRC_REPO/$f"); fi
done
# Normalize each --path to be relative to the repo root (lexically: a symlink inside the repo
# stays a repo path), and refuse paths outside the repo and .git, whose config can hold a token.
for i in "${!PATHS[@]}"; do
  case ${PATHS[$i]} in /*) abs=${PATHS[$i]} ;; *) abs=$SRC_REPO/${PATHS[$i]} ;; esac
  rel=$(realpath -ms --relative-to="$SRC_REPO" "$abs")
  case $rel in
    .|..|../*|.git|.git/*) die "--path takes a file or directory inside the repo, other than .git: ${PATHS[$i]}" ;;
  esac
  PATHS[$i]=$rel
done

SSH_OPTS=(-o BatchMode=yes -o ConnectTimeout=20 -o ServerAliveInterval=30)
[ -z "$JUMP" ] || SSH_OPTS=(-J "$JUMP" "${SSH_OPTS[@]}")
remote() { ssh "${SSH_OPTS[@]}" "$HOST" "$@"; }
# Change the server only on a real copy; print the command otherwise.
remote_change() {
  if [ "$MODE" = copy ]; then remote "$1"; else echo "  [on the server] $1"; fi
}
# --update keeps any file that is newer on the server (a plan ledger or results written there),
# so rerunning this script after working there never overwrites them. zstd: the DSO PLYs
# compress ~2.15x at level 3, which halves a copy over a slow link. --mkpath creates the missing
# destination directories. zstd and --mkpath need rsync >= 3.2.3 at both ends.
sync() {
  local flags=(-ah -s --update --partial --partial-dir=.rsync-partial --mkpath --info=progress2,stats1
               --compress --compress-choice=zstd --compress-level=3)
  [ "$MODE" = dry ] && flags+=(--dry-run)
  rsync "${flags[@]}" -e "ssh ${SSH_OPTS[*]}" "$@"
}
# Symlink $2 -> $1 on the server, refusing to replace a real file or directory.
link_on_server() {
  remote_change "if [ -e '$2' ] && [ ! -L '$2' ]; then echo 'not a symlink, left as is: $2' >&2; exit 1; fi; mkdir -p \"\$(dirname '$2')\" && ln -sfn '$1' '$2'"
}
kb() { du -sLk "$@" 2>/dev/null | awk '{s += $1} END {print s + 0}'; }
gb() { awk -v k="$1" 'BEGIN {printf "%.1f GB", k / 1048576}'; }

# ---- what each tier copies: source, destination on the server, size ----
need_kb=0
plan_line() { printf '  %-58s -> %s  (%s)\n' "$1" "$2" "$(gb "$3")"; need_kb=$((need_kb + $3)); }
show_plan() {
  echo "To $HOST:$REPO${JUMP:+ through $JUMP}. Tiers: ${TIERS[*]:-none}. Paths: ${PATHS[*]:-none}."
  local r d p
  if want core; then
    plan_line "$SRC_DSO/" "$STORE/data/DSO_Dataset/Annotation_Final/" "$(kb "$SRC_DSO")"
    plan_line "data/dso/*.pkl" "$REPO/data/dso/" "$(kb "$SRC_REPO"/data/dso/*.pkl)"
    for r in "${CORE_RUNS[@]}"; do plan_line "work_dirs/$r" "$REPO/work_dirs/$r" "$(kb "$SRC_REPO/work_dirs/$r")"; done
    local ex=(); for d in "${BIG_DUMPS[@]}"; do ex+=(--exclude="$d"); done
    plan_line "$DUMP (without ${BIG_DUMPS[*]})" "$REPO/$DUMP" "$(du -sk "${ex[@]}" "$SRC_REPO/$DUMP" | cut -f1)"
    [ ${#NOTES[@]} -eq 0 ] || plan_line "${NOTES[*]#"$SRC_REPO/"}" "$REPO/" "$(kb "${NOTES[@]}")"
  fi
  if want semantickitti; then
    plan_line "$SRC_SKITTI/" "$STORE/data/SemanticKITTI/" "$(kb "$SRC_SKITTI")"
    plan_line "checkpoint/" "$REPO/checkpoint/" "$(kb "$SRC_REPO/checkpoint")"
    for r in "${SKITTI_RUNS[@]}"; do plan_line "work_dirs/$r" "$REPO/work_dirs/$r" "$(kb "$SRC_REPO/work_dirs/$r")"; done
  fi
  if want analysis; then
    for d in "${BIG_DUMPS[@]}"; do plan_line "$DUMP/$d/" "$STORE/$DUMP/$d/" "$(kb "$SRC_REPO/$DUMP/$d/")"; done
  fi
  for p in "${PATHS[@]}"; do plan_line "$p" "$REPO/$p" "$(kb "$SRC_REPO/$p")"; done
  echo "Total: $(gb "$need_kb")"
}

# ---- checks on this machine ----
need_here() { local p; for p in "$@"; do [ -e "$p" ] || die "missing on this machine: $p"; done; }
if want core; then
  need_here "$SRC_DSO" "$SRC_REPO/data/dso/dso_infos_train.pkl" "$SRC_REPO/$DUMP" \
            "$SRC_REPO/work_dirs/p3former_2xb1_3x_dso/epoch_36.pth"
  for r in "${CORE_RUNS[@]}"; do need_here "$SRC_REPO/work_dirs/$r"; done
fi
if want semantickitti; then
  need_here "$SRC_SKITTI" "$SRC_REPO/checkpoint"
  for r in "${SKITTI_RUNS[@]}"; do need_here "$SRC_REPO/work_dirs/$r"; done
fi
if want analysis; then
  for d in "${BIG_DUMPS[@]}"; do need_here "$SRC_REPO/$DUMP/$d"; done
fi
for p in "${PATHS[@]}"; do need_here "$SRC_REPO/$p"; done
show_plan
[ "$MODE" = plan ] && exit 0

# ---- checks on the server ----
if ! ssh_err=$(remote true 2>&1); then
  echo "Cannot log in to $HOST${JUMP:+ through $JUMP} without a password. ssh said:" >&2
  printf '%s\n' "$ssh_err" | sed 's/^/  /' >&2
  cat >&2 <<EOF
If it is "Permission denied" or "Host key verification failed": authorize this machine's key on
every hop and accept the host keys once (see --help):
  ssh ${JUMP:+-J $JUMP }$HOST true
EOF
  exit 2
fi
if ! remote "test -d '$REPO/.git'"; then
  echo "No git clone at $REPO on $HOST. Clone it there first (see --help)." >&2
  exit 3
fi
avail_kb=$(remote "d='$STORE'; while [ ! -d \"\$d\" ]; do d=\$(dirname \"\$d\"); done; df -Pk \"\$d\" | awk 'NR == 2 {print \$4}'")
echo "Free on the server under $STORE: $(gb "$avail_kb"); needed: $(gb "$need_kb") (less if part of it is already there)"
if [ "$avail_kb" -lt $((need_kb + need_kb / 20)) ] && [ "$FORCE" = 0 ]; then
  echo "Not enough space. Point --store at a bigger disk, or pass --force if part of the data is already there." >&2
  exit 4
fi

# ---- copy ----
if want core; then
  echo "== core: DSO dataset"
  sync "$SRC_DSO/" "$HOST:$STORE/data/DSO_Dataset/Annotation_Final/"
  link_on_server "$STORE/data/DSO_Dataset/Annotation_Final" "$REPO/data/dso/annotations"
  echo "== core: DSO info pkls"
  sync "$SRC_REPO"/data/dso/*.pkl "$HOST:$REPO/data/dso/"
  echo "== core: trained DSO checkpoint and small result logs"
  runs=(); for r in "${CORE_RUNS[@]}"; do runs+=("$SRC_REPO/work_dirs/$r"); done
  sync "${runs[@]}" "$HOST:$REPO/work_dirs/"
  ex=(); for d in "${BIG_DUMPS[@]}"; do ex+=(--exclude="/$(basename "$DUMP")/$d"); done
  sync "${ex[@]}" "$SRC_REPO/$DUMP" "$HOST:$REPO/work_dirs/"
  echo "== core: untracked notes and a pip freeze of the p3former env here"
  [ ${#NOTES[@]} -eq 0 ] || sync "${NOTES[@]}" "$HOST:$REPO/"
  if [ -x "$SRC_PY" ]; then
    mkdir -p "$SRC_REPO/trash"
    "$SRC_PY" -m pip freeze > "$SRC_REPO/$FREEZE"
    sync "$SRC_REPO/$FREEZE" "$HOST:$REPO/trash/"
  else
    echo "No p3former python at $SRC_PY (set P3FORMER_PY), so no pip freeze was copied." >&2
  fi
fi

if want semantickitti; then
  echo "== semantickitti: dataset with its pkls"
  sync "$SRC_SKITTI/" "$HOST:$STORE/data/SemanticKITTI/"
  link_on_server "$STORE/data/SemanticKITTI" "$REPO/data/semantickitti"
  echo "== semantickitti: official checkpoint and the 2xb1 runs"
  sync "$SRC_REPO/checkpoint" "$HOST:$REPO/"
  runs=(); for r in "${SKITTI_RUNS[@]}"; do runs+=("$SRC_REPO/work_dirs/$r"); done
  sync "${runs[@]}" "$HOST:$REPO/work_dirs/"
fi

if want analysis; then
  for d in "${BIG_DUMPS[@]}"; do
    echo "== analysis: $DUMP/$d"
    sync "$SRC_REPO/$DUMP/$d/" "$HOST:$STORE/$DUMP/$d/"
    link_on_server "$STORE/$DUMP/$d" "$REPO/$DUMP/$d"
  done
fi

for p in "${PATHS[@]}"; do
  echo "== path: $p"
  if [ -d "$SRC_REPO/$p" ]; then
    sync "$SRC_REPO/$p/" "$HOST:$REPO/$p/"  # the trailing slash also follows a symlinked directory
  else
    sync --copy-links "$SRC_REPO/$p" "$HOST:$REPO/$(dirname "$p")/"
  fi
done

echo "Done ($MODE)."
if want core; then
  cat <<EOF
Next, on the server:
  - Create the p3former env (DOCs.md, "2026-08-05 — Environment setup"). The versions used here
    are in $REPO/$FREEZE.
  - Check the GPUs: nvidia-smi --query-gpu=name,compute_cap --format=csv. torch 1.10.1+cu111 is
    built up to compute capability 8.6: on 8.9 cuSOLVER fails (.claude/rules/torch-cuda-pitfalls.md),
    and 9.0 (H100, H200) needs a newer torch, with mmcv, spconv and torch_scatter rebuilt for it.
  - Run the tests from the repo root: <the server's p3former python> -m pytest -q tests
EOF
fi
