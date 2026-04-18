#!/usr/bin/env bash
set -e
cd "$(dirname "$0")"
python3 -m uvicorn app.main:app --reload --port 8000
