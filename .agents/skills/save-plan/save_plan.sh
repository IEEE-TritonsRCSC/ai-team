#!/usr/bin/env bash
# save_plan.sh — archive the most recently executed plan into the repo's plans/ dir.
#
# Usage:
#   save_plan.sh <stage-token> <title-slug>
#   save_plan.sh --show            # just print the latest plan path + content, do not copy
#
# <stage-token>  short stage/sub-stage id, e.g. stage2a, stage2c-goalie-balance, stage1
# <title-slug>   concise kebab-case description, e.g. selective-dribbling-reward-curriculum
#
# The destination filename is:
#   plans/<YYYYMMDD>_<HHMMSS>_<stage-token>_<title-slug>.md
# where the date/time are taken from the plan file's last-modified time (when the
# plan was finalised / executed).
set -euo pipefail

PLANS_SRC="${PLANS_SRC:-$HOME/.claude/plans}"

# Repo root: prefer git, else walk up from this script.
if REPO_ROOT="$(git -C "$(dirname "${BASH_SOURCE[0]}")" rev-parse --show-toplevel 2>/dev/null)"; then
  :
else
  REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
fi
DEST_DIR="$REPO_ROOT/plans"

# Find the most recently modified plan file (newest mtime).
LATEST="$(ls -t "$PLANS_SRC"/*.md 2>/dev/null | head -1 || true)"
if [[ -z "${LATEST:-}" ]]; then
  echo "ERROR: no plan files found in $PLANS_SRC" >&2
  exit 1
fi

# Portable mtime -> YYYYMMDD_HHMMSS (BSD/macOS stat first, then GNU stat).
if MTIME="$(stat -f '%Sm' -t '%Y%m%d_%H%M%S' "$LATEST" 2>/dev/null)"; then
  :
else
  MTIME="$(date -r "$(stat -c '%Y' "$LATEST")" '+%Y%m%d_%H%M%S')"
fi

if [[ "${1:-}" == "--show" ]]; then
  echo "LATEST_PLAN: $LATEST"
  echo "PLAN_MTIME:  $MTIME"
  echo "REPO_PLANS:  $DEST_DIR"
  echo "----- PLAN CONTENT -----"
  cat "$LATEST"
  exit 0
fi

if [[ $# -lt 2 ]]; then
  echo "ERROR: need <stage-token> <title-slug> (or --show)" >&2
  echo "Latest plan is: $LATEST (mtime $MTIME)" >&2
  exit 2
fi

STAGE="$1"
TITLE="$2"
mkdir -p "$DEST_DIR"
DEST="$DEST_DIR/${MTIME}_${STAGE}_${TITLE}.md"

cp "$LATEST" "$DEST"
echo "Archived plan:"
echo "  from: $LATEST"
echo "  to:   $DEST"
