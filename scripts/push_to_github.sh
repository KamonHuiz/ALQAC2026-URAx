#!/usr/bin/env bash
# Push this repo to GitHub. Idempotent: safe to run whether or not git is already set up.
# The token is NEVER committed (see .gitignore). Run locally with your GitHub credentials.
#   bash scripts/push_to_github.sh [remote-url]
set -e
REMOTE="${1:-https://github.com/KamonHuiz/ALQAC2026-URAx.git}"

git rev-parse --git-dir >/dev/null 2>&1 || git init
git add -A
git diff --cached --quiet || git commit -m "URAx-LACE: ALQAC 2026 agentic-RAG pipeline

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
git branch -M main
git remote add origin "$REMOTE" 2>/dev/null || git remote set-url origin "$REMOTE"
git push -u origin main
echo "Pushed to $REMOTE"
