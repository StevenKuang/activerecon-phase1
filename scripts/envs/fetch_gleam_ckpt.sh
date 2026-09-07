#!/usr/bin/env bash
# Fetch GLEAM's released checkpoints from the CUHK SharePoint share linked in
# their README.
#
# The share is a *folder* link, and neither of the usual direct-download forms
# works on it: `_layouts/15/download.aspx?share=<id>` returns the HTML viewer
# and `?download=1` returns 403. What does work is the ordinary SharePoint
# REST API — the share link itself grants anonymous read once its cookies are
# in hand, so: GET the share URL to collect cookies, then walk
# `GetFolderByServerRelativeUrl(...)/Files` and pull each blob from
# `GetFileByServerRelativeUrl(...)/$value`.
#
# Upstream calls the 40k-step checkpoint the standard one and notes that the
# stage-2 run excluding the 96 Gibson scenes came out more robust overall, so
# all three 40k checkpoints are fetched (~5 MB each).
set -euo pipefail

OUT=${GLEAM_CKPT_DIR:-$HOME/Projects/GLEAM/ckpt}
SHARE="https://mycuhk-my.sharepoint.com/:f:/g/personal/1155204425_link_cuhk_edu_hk/EiOi5TvbO6JJktArhRGZkLsB8i0ghpwwh-lwFwz4GVASQA?e=Sbhzcm"
BASE="https://mycuhk-my.sharepoint.com/personal/1155204425_link_cuhk_edu_hk"
ROOT="/personal/1155204425_link_cuhk_edu_hk/Documents/ckpt_gleam"
UA="Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/126.0 Safari/537.36"

mkdir -p "$OUT"
JAR=$(mktemp)
trap 'rm -f "$JAR"' EXIT

# The share link's redirect chain is what mints the anonymous-read cookies.
curl -sSL -c "$JAR" -b "$JAR" -A "$UA" -o /dev/null "$SHARE"

for run in train_gleam_stage2 train_gleam_stage2_wo_gibson train_gleam_stage1; do
    dest="$OUT/${run}_40000000_steps.zip"
    if [ -s "$dest" ]; then
        echo "have $dest"
        continue
    fi
    curl -sS -b "$JAR" -A "$UA" -o "$dest" \
        "$BASE/_api/web/GetFileByServerRelativeUrl('$ROOT/$run/models/rl_model_40000000_steps.zip')/\$value"
    # A 403 or the HTML viewer would land here as a "successful" download.
    if [ "$(file -b --mime-type "$dest")" != "application/zip" ]; then
        echo "ERROR: $dest is not a zip (share expired or access revoked?)" >&2
        rm -f "$dest"
        exit 1
    fi
    echo "fetched $dest ($(du -h "$dest" | cut -f1))"
done

echo
echo "Pass one to the adapter, e.g.:"
echo "  --agent-options '{\"checkpoint\": \"$OUT/train_gleam_stage2_40000000_steps.zip\"}'"
