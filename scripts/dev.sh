#!/usr/bin/env bash
set -euo pipefail

# SignaCulture AI - Local Development Launcher
# Starts FastAPI backend and React/Vite frontend.
# Ports are configurable via BACKEND_PORT (default 8000) and FRONTEND_PORT (default 5173).
# Strictly manages and cleans up ONLY its own spawned child PIDs without affecting shared services.

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_ROOT}"

VENV_PYTHON="${REPO_ROOT}/.venv/bin/python3"
if [ ! -x "${VENV_PYTHON}" ]; then
    VENV_PYTHON="python3"
fi

BACKEND_PORT="${BACKEND_PORT:-8000}"
FRONTEND_PORT="${FRONTEND_PORT:-5173}"

# Validation des repertoires requis
if [ ! -d "${REPO_ROOT}/backend/app" ]; then
    echo "[ERROR] backend/app requis mais manquant." >&2
    exit 1
fi

if [ ! -d "${REPO_ROOT}/frontend" ]; then
    echo "[ERROR] frontend requis mais manquant." >&2
    exit 1
fi

# Verification des ports libres (fail fast si occupe pour eviter d'ecraser un service existant)
check_port() {
    local port="$1"
    if command -v nc >/dev/null 2>&1; then
        if nc -z 127.0.0.1 "${port}" 2>/dev/null; then
            echo "[ERROR] Le port ${port} est deja occupe par un autre service. Utilisez BACKEND_PORT ou FRONTEND_PORT pour definir un port alternatif." >&2
            exit 1
        fi
    elif command -v ss >/dev/null 2>&1; then
        if ss -tuln | grep -q ":${port} "; then
            echo "[ERROR] Le port ${port} est deja occupe par un autre service. Utilisez BACKEND_PORT ou FRONTEND_PORT pour definir un port alternatif." >&2
            exit 1
        fi
    fi
}

check_port "${BACKEND_PORT}"
check_port "${FRONTEND_PORT}"

BACKEND_PID=""
FRONTEND_PID=""

cleanup() {
    echo ""
    echo "[DEV] Nettoyage cible des processus enfants crees par ce script..."
    if [ -n "${BACKEND_PID}" ] && kill -0 "${BACKEND_PID}" 2>/dev/null; then
        echo "[DEV] Arret du backend local (PID: ${BACKEND_PID})..."
        kill -TERM "${BACKEND_PID}" 2>/dev/null || true
    fi
    if [ -n "${FRONTEND_PID}" ] && kill -0 "${FRONTEND_PID}" 2>/dev/null; then
        echo "[DEV] Arret du frontend local (PID: ${FRONTEND_PID})..."
        kill -TERM "${FRONTEND_PID}" 2>/dev/null || true
    fi
    wait "${BACKEND_PID}" 2>/dev/null || true
    wait "${FRONTEND_PID}" 2>/dev/null || true
    echo "[DEV] Arret propre termine."
}
trap cleanup EXIT INT TERM

echo "=== Lancement de l'environnement de developpement local SignaCulture AI ==="

# 1. Demarrage du backend FastAPI sur port configurable
echo "[DEV] Demarrage FastAPI backend sur http://127.0.0.1:${BACKEND_PORT}..."
"${VENV_PYTHON}" -m uvicorn app.main:app --app-dir "${REPO_ROOT}/backend" --host 127.0.0.1 --port "${BACKEND_PORT}" &
BACKEND_PID=$!

sleep 1
if ! kill -0 "${BACKEND_PID}" 2>/dev/null; then
    echo "[ERROR] Echec du demarrage du backend FastAPI sur le port ${BACKEND_PORT}." >&2
    exit 1
fi

# 2. Demarrage du frontend Vite avec --strictPort sur port configurable
echo "[DEV] Demarrage Vite frontend sur http://127.0.0.1:${FRONTEND_PORT} (--strictPort)..."
npm --prefix "${REPO_ROOT}/frontend" run dev -- --host 127.0.0.1 --port "${FRONTEND_PORT}" --strictPort &
FRONTEND_PID=$!

sleep 1
if ! kill -0 "${FRONTEND_PID}" 2>/dev/null; then
    echo "[ERROR] Echec du demarrage du frontend Vite sur le port ${FRONTEND_PORT}." >&2
    exit 1
fi

echo "[DEV] Services actifs (Backend: ${BACKEND_PORT}, Frontend: ${FRONTEND_PORT}). Ctrl+C pour quitter."
wait
