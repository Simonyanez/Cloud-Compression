#!/usr/bin/env bash
# Terminal live PDF viewer / text-layout previewer
DOC_DIR="/home/simao/Documents/Repositories/Cloud-Compression/docs/thesis"
TMP_DIR="/tmp/term_preview_$$"
PAGE=${1:-1}

# Temp render MUST live outside DOC_DIR: live_watch.py watches DOC_DIR for .png changes and
# recompiles on any hit. A page_tmp*.png written into DOC_DIR was being picked up as a "source
# changed" event on every preview, causing a recompile per preview and, if previewed faster than
# a compile pass completes (~15-70s for this thesis), a perpetual back-to-back recompile loop.
mkdir -p "$TMP_DIR"
trap 'rm -rf "$TMP_DIR"' EXIT

clear
echo -e "\033[1;34m=== Vista Previa Terminal: main.pdf (Página $PAGE) ===\033[0m"
pdftoppm -png -r 100 -f "$PAGE" -l "$PAGE" "$DOC_DIR/main.pdf" "$TMP_DIR/page_tmp" 2>/dev/null
chafa "$TMP_DIR/page_tmp-$PAGE.png" --size 80x45 2>/dev/null || pdftotext -f "$PAGE" -l "$PAGE" "$DOC_DIR/main.pdf" -
