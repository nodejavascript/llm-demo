#!/usr/bin/env bash
#
# Build, test, publish, purge, smoke-check.
#
# There is no auto-deploy on this repo (unlike nodejavascript.com, which has a
# self-hosted GitHub Actions runner): this script IS the deploy. It builds from
# TypeScript first, so a stale `site/` can never be published by accident.
#
# The `no-store` header the site depends on lives in the Caddy block on the
# droplet, not here — see the README. End-to-end test 1 asserts it is really
# being sent, so a server change that drops it fails loudly.

set -euo pipefail
cd "$(dirname "$0")/.."

echo "== build =="
npm run build

echo "== unit tests =="
node --test test/llm.test.js test/static.test.js test/training.test.js

# The one that would have caught the garbage: it trains the preset and asserts the
# loss clears RECOGNISABLE_LOSS. Only `quick`, because a deploy cannot wait fourteen
# minutes — the full gate is `npm run test:quality`. `quick` alone covers BOTH levels
# now (the README as prose, the built-in names as a list), which is the pair that
# broke in each direction, so a deploy still cannot publish a preset that regresses
# in either of them.
echo "== training quality gate (quick preset, both levels) =="
PRESET=quick node --test test/quality.test.js

echo "== publish =="
rsync -az --delete --rsync-path="sudo rsync" site/ dvs-sites:/srv/llm-demo/
ssh dvs-sites 'sudo chmod -R a+rX /srv/llm-demo'

echo "== purge the CDN =="
( cd ~/Documents/git/gitlab.com/datavisionstudios/docker-compose-master && .venv/bin/python3 ~/.cloudflare_purge.py --all )

echo "== smoke check =="
for path in / /llm.js /app.js /trainer.worker.js /manifest.webmanifest /robots.txt /sitemap.xml /og.png; do
  code=$(curl -s -o /dev/null -w '%{http_code}' "https://llm-demo.nodejavascript.com$path")
  printf '  %-24s %s\n' "$path" "$code"
  [ "$code" = "200" ] || { echo "FAILED: $path returned $code"; exit 1; }
done

cache=$(curl -sI https://llm-demo.nodejavascript.com/llm.js | grep -i '^cache-control' | tr -d '\r')
case "$cache" in
  *no-store*) echo "  cache-control: $cache" ;;
  *) echo "FAILED: /llm.js is not no-store ($cache) — a deploy would be invisible to a returning visitor"; exit 1 ;;
esac

echo
echo "deployed. Run the end-to-end suite against a local server with: npm run test:e2e"
