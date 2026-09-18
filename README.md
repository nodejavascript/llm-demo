# llm-demo

A small GPT — tokenizer, token and position embeddings, causal multi-head
self-attention, a feed-forward block, layer norms, residuals, softmax
cross-entropy, hand-derived backpropagation and AdamW — written from scratch in
**TypeScript** and trained **in the visitor's browser**.

Live: **https://llm-demo.nodejavascript.com** · source: **https://github.com/nodejavascript/llm-demo**

`tsc` compiles five modules to plain JavaScript the browser loads directly: no
bundler, no framework, no runtime dependencies, no server, no API key. The text a
visitor pastes is trained on in a Web Worker on their own machine and is never
uploaded — `test/static.test.js` asserts that no client-side file contains
`fetch`, `XMLHttpRequest`, `sendBeacon`, a WebSocket or an `EventSource`, and the
end-to-end suite pastes a marker string into the page and proves it never appears
in a request or an analytics payload.

It is deliberately small — thousands to hundreds of thousands of parameters. It
demonstrates the mechanism, not the scale, and the page says so.

## Layout

| path | what |
|---|---|
| `src/llm.ts` | the model: tokenizer, kernels, forward, backward, AdamW, sampling |
| `src/trainer-host.ts` | drives a Trainer in time-bounded slices; owns the request/reply types |
| `src/trainer.worker.ts` | module worker wrapper (the page falls back to the main thread) |
| `src/app.ts` | the page: controls, loss chart, analytics, downloads |
| `src/corpora.ts` | the two built-in texts |
| `site/` | **published output** — the compiled `.js` plus the hand-written HTML, CSS, icons and SEO files |
| `test/llm.test.js` | unit: gradients, training, determinism, sampling |
| `test/static.test.js` | guards: page/script agreement, SEO limits, icon bytes, privacy, dead controls |
| `test/e2e.test.js` | end-to-end: a real Chrome against a real server |
| `tools/bench.js` | measured throughput and sample quality per preset |
| `tools/serve.js` | local static server (module workers need a real origin) |
| `tools/deploy.sh` | build → test → rsync → purge → smoke-check |
| `tools/make-icons.py` | regenerates the icon set and the social card |

## Working on it

```bash
npm run build                  # tsc → site/
npm test                       # build + unit + guards   (fast)
npm run test:e2e               # build + the browser suite (~1 min; needs Chrome)
npm run test:all               # both
node tools/bench.js            # what each preset really costs, with samples
node tools/serve.js            # http://127.0.0.1:4320/
python3 tools/make-icons.py    # after changing the mark
npm run deploy                 # publish
```

`npm test` runs the build first, so the suites always test what would ship.
**`site/*.js` is generated — never edit it there.** Change `src/*.ts` and rebuild.

## Tests

**Unit and guards (38).** The load-bearing one is a numerical gradient check of
every parameter array against the analytic gradients. It is the test that makes
the hand-derived backpropagation trustworthy, and it is why the two real bugs
found while building this were caught rather than shipped: an unzeroed backward
buffer (gradients accumulated across batches until the loss went NaN), and every
layer sharing one set of activation buffers (with two layers, the backward pass
for layer 0 read layer 1's numbers). Both guards are themselves guarded — the
suite proves the gradient check *can* fail, and that the dead-control check
catches a button that ships disabled and is never enabled.

**End-to-end (9).** A real Chrome, the real worker, the real download path,
against `tools/serve.js`:

- every asset really arrives `Cache-Control: no-store`;
- the page loads with no console or page errors, one `<h1>`, and the honest
  parameter comparison on screen;
- a full run trains, samples, and downloads the weights and the report under the
  promised filenames;
- stopping early leaves a model that still works, and `top-k = 1` samples greedily;
- **nothing the visitor types leaves the page** — a marker string is pasted into
  the corpus and checked against every request and the whole analytics payload;
- the analytics carry `page_view`, `train_started`, `text_generated`,
  `scroll_depth`, `element_click` and `model_trained` — and a stopped run is *not*
  recorded as a completion;
- no horizontal overflow at seven widths;
- the icon set, manifest, robots and sitemap are all really served.

## 🔴 Why there are no `?v=` tokens, and what replaced them

Cloudflare holds assets at the edge **and in the browser** for four hours
whatever the server sends, so on this site's first deploy the page loaded the
**old `llm.js`** while the fresh HTML beside it showed preset times that had
already been corrected. The first fix was a `?v=` token on every reference —
including the imports inside `app.js`, `trainer-host.js` and the worker.

TypeScript cannot resolve an import specifier with a query on it, so that
mechanism cannot survive the port. The durable fix is the header: **the Caddy
block sets `Cache-Control: no-store` for the whole site**, and end-to-end test 1
asserts every asset really arrives with it. Do not remove that header, and do not
reintroduce a build that needs a token bump to become visible.

## Presets

Step counts are set from measured browser timings, not guesses, and the label
quotes the **cold** first run — the one a visitor actually meets — because the
browser compiles the loops before it settles.

| preset | parameters (names corpus) | steps | measured in Chrome |
|---|---|---|---|
| Quick | 12,821 | 600 | ~14 s cold, ~8 s warm |
| Standard | 75,957 | 250 | ~27 s |
| Thorough | 239,381 | 300 | ~2½ min |

The page shows the rate it is really achieving and an estimate taken from that
rate, so a slower machine says so rather than quietly taking longer.

## Deploy

```bash
npm run deploy     # build, unit tests, rsync to dvs-sites:/srv/llm-demo, purge, smoke check
```

## Licence

The repository is public so the work can be read, reviewed and run. **No licence
is granted**: no permission to reuse, redistribute or incorporate it is given by
its being visible. Adding a licence file is a one-line decision for the owner.
