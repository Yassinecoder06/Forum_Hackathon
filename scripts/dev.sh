#!/usr/bin/env bash
set -euo pipefail

# SignaCulture AI - Local Development Launcher
# Starts FastAPI backend (port 8000) and React/Vite frontend (port 5173).
# Strictly respects docs/INTEGRATION_CONTRACT.md: API localhost:8000, UI localhost:5173.

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_ROOT}"

VENV_PYTHON="${REPO_ROOT}/.venv/bin/python3"
if [ ! -x "${VENV_PYTHON}" ]; then
    VENV_PYTHON="python3"
fi

# Validation des repertoires requis
if [ ! -d "${REPO_ROOT}/backend/app" ]; then
    echo "[ERROR] backend/app requis mais manquant." >&2
    exit 1
fi

if [ ! -d "${REPO_ROOT}/frontend" ]; then
    echo "[ERROR] frontend requis mais manquant." >&2
    exit 1
fi

# Verification des ports libres (fail fast si occupe)
check_port() {
    local port="$1"
    if command -v nc >/dev/null 2>&1; then
        if nc -z 127.0.0.1 "${port}" 2>/dev/null; then
            echo "[ERROR] Le port ${port} est deja occupe." >&2
            exit 1
        fi
    elif command -v ss >/dev/null 2>&1; then
        if ss -tuln | grep -q ":${port} "; then
            echo "[ERROR] Le port ${port} est deja occupe." >&2
            exit 1
        fi
    fi
}

check_port 8000
check_port 5173

BACKEND_PID=""
FRONTEND_PID=""

cleanup() {
    echo ""
    echo "[DEV] Nettoyage et arret des processus..."
    if [ -n "${BACKEND_PID}" ] && kill -0 "${BACKEND_PID}" 2>/dev/null; then
        kill -TERM "${BACKEND_PID}" 2>/dev/null || true
    fi
    if [ -n "${FRONTEND_PID}" ] && kill -0 "${FRONTEND_PID}" 2>/dev/null; then
        kill -TERM "${FRONTEND_PID}" 2>/dev/null || true
    fi
    wait 2>/dev/null || true
    echo "[DEV] Arret termine."
}
trap cleanup EXIT INT TERM

echo "=== Lancement de l'environnement de developpement local SignaCulture AI ==="

# 1. Demarrage du backend FastAPI
echo "[DEV] Demarrage FastAPI backend sur http://127.0.0.1:8000..."
"${VENV_PYTHON}" -m uvicorn app.main:app --app-dir "${REPO_ROOT}/backend" --host 127.0.0.1 --port 8000 &
BACKEND_PID=$!

# Verification immediate que le backend demarre
sleep 1
if ! kill -0 "${BACKEND_PID}" 2>/dev/null; then
    echo "[ERROR] Echec du demarrage du backend FastAPI." >&2
    exit 1
fi

# 2. Demarrage du frontend Vite avec --strictPort
echo "[DEV] Demarrage Vite frontend sur http://127.0.0.1:5173 (--strictPort)..."
npm --prefix "${REPO_ROOT}/frontend" run dev -- --host 127.0.0.1 --port 5173 --strictPort &
FRONTEND_PID=$!

sleep 1
if ! kill -0 "${FRONTEND_PID}" 2>/dev/null; then
    echo "[ERROR] Echec du demarrage du frontend Vite." >&2
    exit 1
fi

echo "[DEV] Services actifs (Backend: 8000, Frontend: 5173). Ctrl+C pour quitter."
wait
