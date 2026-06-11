#!/bin/sh
set -e

: "${REDIS_CONNECTION_STRING:?REDIS_CONNECTION_STRING must be set (see .env.example)}"

if [ ! -f data/registry.db ]; then
  echo "data/registry.db not found — build it with the training pipeline first." >&2
  exit 1
fi

exec feast serve -h 0.0.0.0 -p 6566
