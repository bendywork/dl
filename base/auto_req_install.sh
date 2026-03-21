#!/usr/bin/env bash
set -euo pipefail

############################################################
# Auto scan all sibling chapter dirs for .py / .ipynb,
# generate requirements.txt, and install only missing deps.
#
# Directory layout assumed:
#   project_root/
#   ├── chapter_01/
#   ├── chapter_02/
#   └── this_script_dir/   ← script lives here
############################################################

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SCRIPT_DIRNAME="$(basename "$SCRIPT_DIR")"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
OUTPUT_FILE="$SCRIPT_DIR/requirements.txt"

echo "=================================================="
echo " 📦  Dependency Scanner"
echo "=================================================="
echo "[INFO] Script dir   : $SCRIPT_DIR"
echo "[INFO] Project root : $PROJECT_ROOT"
echo "[INFO] Output file  : $OUTPUT_FILE"
echo

# ── 1. Ensure pipreqs ─────────────────────────────────────────────────────────
if ! command -v pipreqs &>/dev/null; then
  echo "[INFO] pipreqs not found, installing..."
  pip install pipreqs --quiet
fi

# ── 2. Collect sibling chapter dirs (exclude script's own dir) ────────────────
CHAPTER_DIRS=()
for d in "$PROJECT_ROOT"/*/; do
  [[ "$(basename "${d%/}")" == "$SCRIPT_DIRNAME" ]] && continue
  [[ -d "$d" ]] && CHAPTER_DIRS+=("${d%/}")
done

if [[ ${#CHAPTER_DIRS[@]} -eq 0 ]]; then
  echo "[WARN] No sibling chapter directories found under $PROJECT_ROOT"
  exit 0
fi

echo "[INFO] Directories to scan:"
for d in "${CHAPTER_DIRS[@]}"; do
  echo "         → $d"
done
echo

# ── 3. Generate requirements.txt via pipreqs ──────────────────────────────────
# pipreqs doesn't accept multiple paths, so scan project root and
# ignore the script's own directory with --ignore.
echo "[INFO] Running pipreqs (this may take a moment)..."
pipreqs "$PROJECT_ROOT" \
  --force \
  --encoding=utf-8 \
  --mode no-pin \
  --savepath "$OUTPUT_FILE" \
  --ignore "$SCRIPT_DIRNAME"

echo
echo "================ Generated requirements.txt ================="
if [[ -s "$OUTPUT_FILE" ]]; then
  cat "$OUTPUT_FILE"
else
  echo "(empty)"
fi
echo "============================================================="
echo

# ── 4. Check which packages are already installed ────────────────────────────
if [[ ! -s "$OUTPUT_FILE" ]]; then
  echo "[INFO] No external dependencies detected. Nothing to install."
  exit 0
fi

MISSING=()
PRESENT=()

while IFS= read -r line; do
  # Skip blank lines and comments
  [[ -z "$line" || "$line" == \#* ]] && continue

  # Normalize: strip version specifiers, lowercase, unify - and _
  # pipreqs --mode no-pin outputs bare names, but guard anyway
  pkg_raw=$(echo "$line" | sed 's/[>=<!\[].*//')
  pkg_key=$(echo "$pkg_raw" | tr '[:upper:]' '[:lower:]' | tr '-' '_')

  # pip show handles name normalisation internally
  if pip show "$pkg_key" &>/dev/null 2>&1; then
    PRESENT+=("$pkg_raw")
  else
    MISSING+=("$line")
  fi
done < "$OUTPUT_FILE"

# ── 5. Report and install only what is missing ───────────────────────────────
if [[ ${#PRESENT[@]} -gt 0 ]]; then
  echo "[SKIP] Already installed ($(( ${#PRESENT[@]} )) pkg(s)): ${PRESENT[*]}"
fi

if [[ ${#MISSING[@]} -eq 0 ]]; then
  echo
  echo "✅ All dependencies are satisfied. Nothing to install."
else
  echo "[INFO] Missing  ($(( ${#MISSING[@]} )) pkg(s)): ${MISSING[*]}"
  echo
  echo "=================================================="
  echo " 🚀 Installing ${#MISSING[@]} missing package(s)..."
  echo "=================================================="
  # Feed only the missing lines to pip — no reinstall, no version clash
  printf '%s\n' "${MISSING[@]}" | pip install -r /dev/stdin
fi

echo
echo "=================================================="
echo " ✅ Done!"
echo "=================================================="
