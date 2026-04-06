#!/bin/bash
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# ── Install Ollama ───────────────────────────────────────────────────────
if ! command -v ollama &>/dev/null; then
    echo "Error: Ollama is not installed. Download it from https://ollama.com"
    exit 1
fi

if ! ollama list &>/dev/null; then
    echo "Error: Ollama daemon is not running. Start it with: ollama serve"
    exit 1
fi

# ── Install dependencies ──────────────────────────────────────────────────────
echo "Installing dependencies..."
pip install -r requirements.txt -q

# ── Launch UI ─────────────────────────────────────────────────────────────────
echo "Starting Data Analyst UI at http://localhost:8501"
streamlit run app.py
