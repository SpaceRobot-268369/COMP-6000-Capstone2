#!/bin/sh
set -eu

cd /app

if [ -f package-lock.json ]; then
  rm -f package-lock.json
fi

if [ -d node_modules ]; then
  rm -rf node_modules/* node_modules/.[!.]* node_modules/..?* 2>/dev/null || true
fi

npm install
exec "$@"
