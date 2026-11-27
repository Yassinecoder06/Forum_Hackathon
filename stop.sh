#!/usr/bin/env bash
# Stop the SignePourMoi presentation started by ./start.sh: the deck :5215 and the interpreter :5207.
# The API :8040 (shared with the Studio) is stopped only if start.sh started it; ./stop.sh --keep-api leaves it running.
set -u
STATE=$HOME/.cache/signepourmoi
KEEP_API=0; [ "${1:-}" = "--keep-api" ] && KEEP_API=1

# the process group of whatever listens on a port, only if its command line matches what we expect
stop_port() {  # port, expected command pattern, name
  local pid pgid
  pid=$(ss -ltnpH "sport = :$1" | grep -o 'pid=[0-9]*' | head -1 | cut -d= -f2)
  if [ -z "$pid" ]; then echo "$3 $1: not running"; return; fi
  if ! tr '\0' ' ' < /proc/$pid/cmdline | grep -qE "$2"; then echo "$3 $1: something else listens there, left alone"; return; fi
  pgid=$(ps -o pgid= -p "$pid" | tr -d ' ')
  kill -- -"$pgid" 2>/dev/null || kill "$pid"
  for _ in $(seq 1 10); do ss -ltnH "sport = :$1" | grep -q . || break; sleep 1; done
  echo "$3 $1: stopped"
}

systemctl --user stop presentation-finale 2>/dev/null
stop_port 5215 "http.server 5215" "deck"
stop_port 5207 "FH-presentation/frontend/.*vite.*--port 5207" "interpreter"

if [ "$KEEP_API" = 1 ]; then
  echo "API 8040: kept (--keep-api)"
elif [ -f "$STATE/api-started-by-us" ]; then
  stop_port 8040 "api\.py" "API"
  rm -f "$STATE/api-started-by-us"
else
  echo "API 8040: not started by start.sh, left running"
fi
