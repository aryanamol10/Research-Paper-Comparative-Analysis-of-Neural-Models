#!/usr/bin/env bash
# Bundles the paper into a zip you can upload with Overleaf -> New Project -> Upload Project.
# Usage (from repo root or paper/):  bash paper/scripts/package_overleaf.sh [venue]
#   venue = one of the file names in paper/venues/ without .tex (optional; default keeps venue.tex)
set -euo pipefail
cd "$(dirname "$0")/.."

VENUE="${1:-}"
OUT="overleaf_${VENUE:-default}.zip"
STAGE="$(mktemp -d)"

cp -R main.tex isec_k12_abstract.tex venue.tex references.bib sections tables venues "$STAGE"/
mkdir -p "$STAGE/figures"
cp figures/*.pdf figures/*.tex "$STAGE/figures/"
if [[ -n "$VENUE" ]]; then
  cp "venues/$VENUE.tex" "$STAGE/venue.tex"
fi

rm -f "$OUT"
(cd "$STAGE" && zip -qr "$OLDPWD/$OUT" .)
rm -rf "$STAGE"
echo "wrote paper/$OUT"
