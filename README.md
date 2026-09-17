# llm-demo

A small GPT — tokenizer, token and position embeddings, causal multi-head
self-attention, a feed-forward block, layer norms, residuals, softmax
cross-entropy, hand-derived backpropagation and AdamW — written from scratch in
plain JavaScript and trained **in the visitor's browser**.

Live: **https://llm-demo.nodejavascript.com**

No framework, no build step, no server, no API key, no dependencies. The text a
visitor pastes is trained on in a Web Worker on their own machine and is never
uploaded; `test/static.test.js` asserts that no client-side file contains
`fetch`, `XMLHttpRequest`, `sendBeacon`, a WebSocket or an `EventSource`.

It is deliberately small — thousands to hundreds of thousands of parameters. It
demonstrates the mechanism, not the scale, and the page says so.

## Files

| path | what |
|---|---|
| `site/llm.js` | the model: tokenizer, kernels, forward, backward, AdamW, sampling |
| `site/trainer-host.js` | drives a Trainer in time-bounded slices; used by both the worker and the main thread |
| `site/trainer.worker.js` | module worker wrapper (the page falls back to the main thread if module workers are unavailable) |
| `site/app.js` | the page: controls, loss chart, analytics, downloads |
| `site/corpora.js` | the two built-in texts |
| `tools/bench.js` | measured throughput and sample quality per preset |
| `tools/serve.js` | a local static server (module workers need a real origin) |
| `tools/make-icons.py` | regenerates the icon set and the social card |

## Working on it

```bash
npm test                       # 38 tests: gradients, training, sampling, page guards
node tools/bench.js            # what each preset really costs, with samples
node tools/serve.js            # http://127.0.0.1:4320/
python3 tools/make-icons.py    # after changing the mark
```

## 🔴 Bump the `?v=` token whenever you change anything

Assets are addressed as `./llm.js?v=2` — **including the imports inside `app.js`,
`trainer-host.js` and the `new Worker(...)` call**, not just the `<script>` tag.
Cloudflare holds assets at the edge and in the browser for four hours whatever the
server sends, so a deploy with an unchanged token is invisible to anyone who has
already visited; that is exactly what happened on this site's first deploy, where
the page loaded the old `llm.js` and showed preset times that had already been
corrected. A test fails if any asset is loaded without a token or if two assets
carry different ones.

`npm test` includes a numerical gradient check of every parameter array against
the analytic gradients. It is the test that makes the hand-derived backprop
trustworthy, and it is why the two real bugs found while building this — an
unzeroed backward buffer and every layer sharing one set of activation buffers —
were caught rather than shipped.

## Presets

Step counts are set from measured browser timings, not guesses. Re-measure with
`tools/bench.js` after changing anything in the model.

| preset | parameters (names corpus) | steps | measured in Chrome, warm |
|---|---|---|---|
| Quick | 12,821 | 600 | ~8 s (14 s on the very first run in a fresh tab) |
| Standard | 75,957 | 250 | ~27 s |
| Thorough | 239,381 | 300 | ~2 min |

The page also shows the rate it is really achieving and an estimate taken from
that rate, so a slower machine says so instead of quietly taking longer.

## Deploy

```bash
rsync -az --delete --rsync-path="sudo rsync" site/ dvs-sites:/srv/llm-demo/
ssh dvs-sites 'sudo systemctl reload caddy'
.venv/bin/python3 ~/.cloudflare_purge.py --all      # from the dcm repo
```

Caddy serves `/srv/llm-demo` for `llm-demo.nodejavascript.com` with
`Cache-Control: no-store` on `robots.txt` and `sitemap.xml`. **`site/` is
published wholesale by `rsync --delete`** — nothing goes in that folder unless it
should be on the web, and a test asserts there are no stray files in it.
