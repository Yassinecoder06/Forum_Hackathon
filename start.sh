#!/usr/bin/env bash
# Launch the SignePourMoi presentation (http://localhost:5215/):
#   1. API :8040 (workspace/show) and the models the presentation uses: Whisper, C0 + refiners, emotion2vec.
#      No Gemma brain: the presentation never calls it (only /live, /analyze-audio, /lsf/generate), so it is switched off
#      (no search for landolsi :8090, nothing sent there) and neither the brain nor B2 is warmed.
#   2. fast live interpreter :5207 (Vite, .worktrees/FH-presentation, /api -> 8040)
#   3. the deck :5215 (apps/presentation)
# Anything already listening is left as it is. Stop with ./stop.sh.
set -u
ROOT="$(cd "$(dirname "$0")" && pwd)"            # the repo (paths below are relative to it, see STRUCTURE.md)
SHOW=$ROOT/workspace/show/run
INTERP=$ROOT/.worktrees/FH-presentation
DECK=$ROOT/apps/presentation
STATE=$HOME/.cache/signepourmoi && mkdir -p "$STATE"
up() { ss -ltnH "sport = :$1" | grep -q .; }

# 1. API + models
if up 8040; then
  echo "API 8040: already running"
else
  echo "API 8040: starting"
  # the show env without its brain lines (search for Gemma on :8090), then the brain pointed at a closed local port
  # exec: the server replaces the sub-shell and owns only its log, so nothing keeps this script's output open
  ( cd "$SHOW" && . <(grep -v ':8090' ./env.sh) \
      && export SIGNACULTURE_LLM_API=openai SIGNACULTURE_LLM_URL=http://127.0.0.1:9 \
      && exec setsid nohup "$ROOT/workspace/.venv/bin/python" api.py > api.log 2>&1 < /dev/null ) &
  echo $! > "$SHOW/api.pid"
  touch "$STATE/api-started-by-us"          # stop.sh stops the API only if we started it
  for _ in $(seq 1 60); do curl -sf -o /dev/null http://127.0.0.1:8040/health && break; sleep 2; done
  curl -sf -o /dev/null http://127.0.0.1:8040/health || { echo "API did not answer: see $SHOW/api.log"; exit 1; }
  echo "models: loading (Whisper, C0 + refiners, emotion2vec)"
  A=http://127.0.0.1:8040; t() { curl -s -m 600 -o /dev/null -w "%{http_code} %{time_total}s" "$@"; }
  echo "  Whisper + C0:  $(t -X POST "$A/live/warmup?engine=c0")"
  for s in "Bonjour, je voudrais un billet pour Paris." "Demain il fera beau et nous irons au parc."; do
    echo "  C0 sentence:   $(t -X POST $A/lsf/generate -H 'Content-Type: application/json' -d "{\"text\":\"$s\",\"engine\":\"c0\"}")"; done
  echo "  Whisper clip:  $(t -F file=@$SHOW/warm/bonjour.wav $A/live/transcribe)"
  echo "  emotion2vec:   $(t -F file=@$SHOW/warm/bonjour.wav -F 'cues=[{"start_ms":0,"end_ms":2400}]' $A/analyze-audio/emotion-segments)"
fi

# 2. fast interpreter
if up 5207; then
  echo "interpreter 5207: already running"
else
  echo "interpreter 5207: starting"
  ( cd "$INTERP/frontend" && SIGNACULTURE_API_TARGET=http://127.0.0.1:8040 SIGNACULTURE_HOME_PAGE="/interprete/?rapide=1" \
      exec setsid nohup npx vite --host 0.0.0.0 --port 5207 --strictPort > "$INTERP/run/rapide.log" 2>&1 < /dev/null ) &
fi

# 3. deck
if up 5215; then
  echo "deck 5215: already running"
else
  echo "deck 5215: starting"
  systemctl --user reset-failed presentation-finale 2>/dev/null
  systemd-run --user --unit presentation-finale --collect "$DECK/start.sh" > /dev/null
fi

# wait for the pages
for p in 5207 5215; do for _ in $(seq 1 20); do up $p && break; sleep 1; done; done
for p in 8040 5207 5215; do printf "%s %s\n" "$p" "$(up $p && echo up || echo DOWN)"; done
echo
echo "Presentation: http://localhost:5215/"
