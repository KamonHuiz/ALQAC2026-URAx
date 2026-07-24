#!/usr/bin/env bash
# Helper to push this repo to GitHub. Run locally after reviewing the files.
# The token is NEVER committed (see .gitignore). Set the remote to your repo.
set -e
REMOTE="${1:-https://github.com/KamonHuiz/ALQAC2026-URAx.git}"

git init
git add .
git commit -m "URAx-LACE: full ALQAC 2026 agentic-RAG pipeline

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
git branch -M main
git remote add origin "$REMOTE" 2>/dev/null || git remote set-url origin "$REMOTE"
git push -u origin main
echo "Pushed to $REMOTE"
