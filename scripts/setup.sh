#!/usr/bin/env bash
set -euo pipefail

# SignaCulture AI - Local Environment Setup
# Installs backend runtime + development dependencies and frontend packages.

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_ROOT}"

echo "=== SignaCulture AI - Setup des environnements locaux ==="

# 1. Verification des pre-requis systeme
command -v python3 >/dev/null 2>&1 || { echo "[ERROR] python3 requis mais non trouve." >&2; exit 1; }
echo "[OK] python3 detecte: $(python3 --version)"

command -v node >/dev/null 2>&1 || { echo "[ERROR] node requis pour le frontend mais non trouve." >&2; exit 1; }
echo "[OK] node detecte: $(node --version)"
command -v npm >/dev/null 2>&1 || { echo "[ERROR] npm requis mais non trouve." >&2; exit 1; }
echo "[OK] npm detecte: $(npm --version)"

# 2. Configuration du virtualenv Python (.venv)
VENV_DIR="${REPO_ROOT}/.venv"
if [ ! -d "${VENV_DIR}" ]; then
    echo "[SETUP] Creation du virtualenv dans ${VENV_DIR}..."
    if command -v uv >/dev/null 2>&1; then
        uv venv "${VENV_DIR}"
    else
        python3 -m venv "${VENV_DIR}"
    fi
else
    echo "[OK] Virtualenv existant: ${VENV_DIR}"
fi

VENV_PYTHON="${VENV_DIR}/bin/python3"

# 3. Installation des dependances backend (runtime + dev obligatoires)
echo "[SETUP] Installation des dependances Python..."
if command -v uv >/dev/null 2>&1; then
    if [ -f "${REPO_ROOT}/backend/requirements.txt" ]; then
        uv pip install -r "${REPO_ROOT}/backend/requirements.txt" --python "${VENV_PYTHON}"
    fi
    if [ -f "${REPO_ROOT}/requirements-dev.txt" ]; then
        uv pip install -r "${REPO_ROOT}/requirements-dev.txt" --python "${VENV_PYTHON}"
    elif [ -f "${REPO_ROOT}/backend/requirements-dev.txt" ]; then
        uv pip install -r "${REPO_ROOT}/backend/requirements-dev.txt" --python "${VENV_PYTHON}"
    else
        uv pip install pytest httpx uvicorn fastapi numpy --python "${VENV_PYTHON}"
    fi
else
    if [ -f "${REPO_ROOT}/backend/requirements.txt" ]; then
        "${VENV_PYTHON}" -m pip install -r "${REPO_ROOT}/backend/requirements.txt"
    fi
    if [ -f "${REPO_ROOT}/requirements-dev.txt" ]; then
        "${VENV_PYTHON}" -m pip install -r "${REPO_ROOT}/requirements-dev.txt"
    elif [ -f "${REPO_ROOT}/backend/requirements-dev.txt" ]; then
        "${VENV_PYTHON}" -m pip install -r "${REPO_ROOT}/backend/requirements-dev.txt"
    else
        "${VENV_PYTHON}" -m pip install pytest httpx uvicorn fastapi numpy
    fi
fi
echo "[OK] Dependances Python installees."

# 4. Installation des dependances frontend
echo "[SETUP] Installation des dependances frontend..."
if [ -d "${REPO_ROOT}/frontend" ]; then
    cd "${REPO_ROOT}/frontend"
    npm install
    cd "${REPO_ROOT}"
    echo "[OK] Dependances frontend installees."
else
    echo "[ERROR] Repertoire frontend requis dans le depot final." >&2
    exit 1
fi

echo "=== Setup termine avec succes ==="
