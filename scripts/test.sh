#!/usr/bin/env bash
set -euo pipefail

# SignaCulture AI - Unified Test Suite Runner
# Executes schema validation tests, backend unit tests, experiments tests, frontend lint & build, and optional smoke tests.

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_ROOT}"

VENV_PYTHON="${REPO_ROOT}/.venv/bin/python3"
if [ ! -x "${VENV_PYTHON}" ]; then
    VENV_PYTHON="python3"
fi

API_URL="${API_URL:-http://127.0.0.1:8000}"
RUN_SMOKE="${RUN_SMOKE:-false}"

echo "=== SignaCulture AI - Suite de Tests Complète ==="

# 1. Tests des schemas et contrats partages (backend/app/schemas/test_models.py)
if [ ! -d "${REPO_ROOT}/backend/app/schemas" ]; then
    echo "[ERROR] Repertoire backend/app/schemas requis mais manquant." >&2
    exit 1
fi
echo "[TEST] 1/5 - Execution des tests de validation des schemas et modeles..."
"${VENV_PYTHON}" -m pytest "${REPO_ROOT}/backend/app/schemas" -v

# 2. Tests unitaires Backend
if [ ! -d "${REPO_ROOT}/backend/tests" ]; then
    echo "[ERROR] Repertoire backend/tests requis mais manquant." >&2
    exit 1
fi
echo "[TEST] 2/5 - Execution des tests unitaires backend..."
"${VENV_PYTHON}" -m pytest "${REPO_ROOT}/backend/tests" -v

# 3. Tests des experimentations (motion extraction)
if [ ! -d "${REPO_ROOT}/experiments/motion_extraction" ]; then
    echo "[ERROR] Repertoire experiments/motion_extraction requis mais manquant." >&2
    exit 1
fi
echo "[TEST] 3/5 - Execution des tests experiments/motion_extraction..."
"${VENV_PYTHON}" -m pytest "${REPO_ROOT}/experiments/motion_extraction" -v

# 4. Frontend Lint & Build
if [ ! -d "${REPO_ROOT}/frontend" ]; then
    echo "[ERROR] Repertoire frontend requis mais manquant." >&2
    exit 1
fi
echo "[TEST] 4/5 - Execution frontend lint et build..."
npm --prefix "${REPO_ROOT}/frontend" run lint
npm --prefix "${REPO_ROOT}/frontend" run build

# 5. Tests d'integration et Smoke Tests (optionnels via RUN_SMOKE=true ou make smoke)
if [ "${RUN_SMOKE}" = "true" ]; then
    echo "[TEST] 5/5 - Execution des smoke tests contre ${API_URL}..."
    "${VENV_PYTHON}" "${REPO_ROOT}/scripts/smoke_test.py" --base-url "${API_URL}"
else
    echo "[INFO] 5/5 - Smoke tests ignores par defaut (optionnels via RUN_SMOKE=true ou 'make smoke' avec serveur actif)."
fi

echo "=== Tous les tests ont ete executes avec succes ==="
