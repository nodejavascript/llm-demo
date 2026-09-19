# llm-demo.nodejavascript.com

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
| `src/app.ts` | the page: controls, loss chart, downloads |
| `src/consent.ts` | the cookie gate — the only file that may load Google's script, and only after a yes |
| `src/corpora.ts` | the two built-in texts |
| `site/` | **published output** — the compiled `.js` plus the hand-written HTML, CSS, icons and SEO files |
| `test/llm.test.js` | unit: gradients, training, determinism, sampling |
| `test/static.test.js` | guards: page/script agreement, SEO limits, icon bytes, privacy, dead controls |
| `test/training.test.js` | the fast guarantee: each preset is trained long enough, and sampling truncates by default |
| `test/quality.test.js` | the slow guarantee: trains each preset and asserts the words are in the right order, not just the right set |
| `test/e2e.test.js` | end-to-end: a real Chrome against a real server |
| `tools/bench.js` | measured throughput and sample quality per preset |
| `tools/probe-prose.mjs` | what it does with prose (a diary) rather than with a list — the honest limit |
| `tools/probe-defaults.mjs` | what the sample controls do to the same trained model |
| `tools/probe-curve.mjs` | how many steps a shape needs before real words appear |
| `tools/serve.js` | local static server (module workers need a real origin) |
| `tools/deploy.sh` | build → test → rsync → purge → smoke-check |
| `tools/verify-live-consent.mjs` | the same gate checked in a browser against the deployed site |
| `tools/make-icons.py` | regenerates the icon set and the social card |

## Working on it

```bash
npm run build                  # tsc → site/
npm test                       # build + unit + guards + the fast training checks (~1 s)
npm run test:quality           # build + the training guarantee — slow: ~14 min for both
                               #   levels of all three presets, or PRESET=quick ≈ 70 s
npm run test:e2e               # build + the browser suite (~1 min; needs Chrome)
npm run test:all               # everything
node tools/bench.js            # what each preset really costs, with samples
node tools/serve.js            # http://127.0.0.1:4320/
python3 tools/make-icons.py    # after changing the mark
npm run deploy                 # publish
```

`npm test` runs the build first, so the suites always test what would ship.
**`site/*.js` is generated — never edit it there.** Change `src/*.ts` and rebuild.

## Tests

**Unit and guards (51).** The load-bearing one is a numerical gradient check of
every parameter array against the analytic gradients. It is the test that makes
the hand-derived backpropagation trustworthy, and it is why the two real bugs
found while building this were caught rather than shipped: an unzeroed backward
buffer (gradients accumulated across batches until the loss went NaN), and every
layer sharing one set of activation buffers (with two layers, the backward pass
for layer 0 read layer 1's numbers). Both guards are themselves guarded — the
suite proves the gradient check *can* fail, and that the dead-control check
catches a button that ships disabled and is never enabled.

**The training guarantee (7, and the half that matters is slow).** Everything
above asserts *mechanism* — that the gradients are right, that the vocabulary
round-trips, that the shapes compose. All of it passed while the shipped presets
were producing word-shaped noise, because nothing checked that the model had
trained **enough to be worth sampling from**.

Two files assert *outcome* instead. `test/training.test.js` holds the
configuration to the measurement: each preset's step count must be at or above the
number its shape is measured to need; a dearer preset must cost more *work* (steps ×
parameters); the advertised seconds must be arithmetically possible at the measured
rate; `sample()` must truncate by default, asserted **by identity** so that with a
fixed seed the no-argument call equals the explicit good defaults; the page's top-k
slider must not default to 0; and `plannedSteps()` must return the preset's full
step count on a realistic corpus.

That last one exists because **the cap was the second half of the same bug**. The
host capped every run at 40 passes over the corpus — computed from the *corpus*,
not the preset — so on the built-in 4,359-character name list every preset ran
exactly **605 steps**, and raising `quick` from 600 to 3,000 would have changed
nothing at all. No unit test saw it, because the unit tests drive `new Trainer()`
directly and never went through the host that applies the cap. It now lives in one
exported function, `plannedSteps()`, used by the host and by the tests.

**And a loss is a sample, not a value.** The trainer starts from a random
initialisation, so the same shape at the same step count lands somewhere different
every run: one configuration measured **1.310** once and **1.679** another time.
That is why the quality gate trains with a **fixed seed**, and why the seed is
fixed to an arbitrary round number rather than one picked to flatter a preset.

### 🔴 The tokeniser level follows the text — and the two levels want OPPOSITE things

The site was character level, and character level produced *"thanger
thand-wrardeady"* on prose. That is not a training problem and no amount of extra
training fixes it, because a character model never learns a word as a unit. Measured
with the same `Trainer`, the same preset and the same 1,500 steps — the tokeniser was
the only thing that differed:

| | loss | output |
|---|---|---|
| character level | 1.278 | *"itical web automatily, and Go, AnfuxP stomatttts"* |
| **word level** | **0.081** | *"FIELDER / Senior Software Engineer, Full Stack / Hamilton, Ontario, Canada / <contact address>"* |

So it was switched to words. **And that broke the other half of the site**, which the
measurement caught: the DEFAULT corpus is a list of six hundred given names, and on a
list of one-off items a word-level model can do nothing but recite — every name is a
single token occurring once, so it emitted `Abigail Adam Adrian Aiden` forever and
invented nothing at all.

Both were true, so the level now follows the text, decided by `detectLevel()` on the
share of distinct words:

| | words | distinct | ratio | level |
|---|---|---|---|---|
| the built-in name list | 653 | 653 | **1.00** | character |
| the built-in dialogue | 522 | 180 | 0.34 | word |

A list where almost every item occurs once is tokenised by character, so the model can
compose new entries from real letters. Prose is tokenised by word, so it reads as
language. Texts under 200 words stay at word level, because a ratio from a hundred
tokens is not evidence of anything.

#### 🔴 On a list, MORE TRAINING IS WORSE — measured

This is the most counter-intuitive thing here, and it is asserted in the test suite so
nobody "fixes" it later. At character level on the 653-name list:

| preset | steps | time | loss | new names, of 36 lines |
|---|---|---|---|---|
| **`quick`** | 4,000 | 52 s | 0.810 | **29 — and 26 name-shaped** |
| `standard` | 1,500 | 139 s | 0.572 | 26, one repeated four times |
| `thorough` | 3,000 | 293 s | 0.364 | **16** — reciting, with stutters like `Molll y`, `Moseses` |

The loss keeps falling while the thing the visitor wants disappears: a bigger model on
a small list simply memorises it faster. **`quick` is the best preset for a list and
`thorough` is the worst**, which is why each character-level chip carries a note saying
so — a visitor who spends five minutes and gets worse output would otherwise read it
as a broken demo.

#### The metric, and why there are two of them

The two levels fail differently, so one number cannot cover both. `COHERENCE_FLOOR`
in `src/llm.ts` holds both, and the live page judges its own output with the same
constant the build asserts against:

- **word level** — the share of a sample's adjacent word PAIRS that occur in the
  corpus. Every token is already a real word, so counting words says nothing; what
  separates text from a bag of words is the ORDER. Floor **0.95**.
- **character level** — the share of a sample's 5-letters-or-longer words that occur
  anywhere inside the corpus. This is a *proxy*, not a score of how many entries are
  "real": a word-shaped run passes it and random letter salad fails it, which is
  exactly the discrimination needed. Floor **0.12**.

Measured on `README.md` (11,895 characters, 1,945 word tokens, 939-word vocabulary):

| steps | `quick` loss · pairs | `standard` loss · pairs |
|---|---|---|
| 100 | 3.189 · 86% | 2.620 · 91% |
| 250 | 1.048 · 97% | 0.451 · 98% |
| 500 | 0.235 · 100% | 0.127 · 100% |
| 1000 | 0.131 · 100% | 0.095 · 100% |

Under-trained word output sits at 86-91% — an assortment of real words. `test/quality.test.js`
trains **each preset, at each level the site ships, for the run the page would really
take** — six runs in all, three presets × two levels — through the same
`plannedSteps()` — and asserts the smoothed loss is below
`RECOGNISABLE_LOSS[level]` *and* that the level's own coherence is above its floor. It
also **reports** how much of each sample is copied from the corpus, because on corpora
this size the answer is "a lot" and a report that hid it would be the dishonest
version. `tools/deploy.sh` runs the `quick` preset on every deploy, so a preset that
regresses cannot be published.

**End-to-end (17).** A real Chrome, the real worker, the real download path,
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
- **the cookie gate**, which is the only part of this site that can be wrong in a
  way nobody can see: with no answer, a refusal, or `?ga=off`, the request log is
  checked for **zero** requests to Google and `window.gtag` must be `undefined`; a
  yes must fetch the tag and open the event route; a refusal must not come back;
  the switch must start off; and turning it off must reload into a page that
  contacts nobody;
- no horizontal overflow at seven widths, and the banner never covering the footer;
- the icon set, manifest, robots and sitemap are all really served.

**Live (13 checks).** `node tools/verify-live-consent.mjs` runs the same gate
against the deployed site in five states, which is where the DNS, the TLS, the
Caddy headers and the cache purge all have to be right at once.

## Cookies — the tag is not in the page, and that is the point

`site/index.html` contains **no Google script and no analytics code at all**. The
measurement id rides on the consent script as an attribute (`data-ga-id`), and
`site/consent.js` appends the tag **only after a yes** — so a visit that refuses
makes no request to Google and receives no cookie. The choice is stored in
`localStorage` (`analytics_consent`), not in a cookie, and the address parameters
`?ga=off` / `?ga=on` override it, so whoever runs the site can keep their own visits
out of the count.

This is the same model as `inputresponse.com`, and it exists because the previous
arrangement — tag in the head, opt-out in front of it — is not consent: the
script had already loaded and the cookie was already set before anyone was asked.

If you are changing anything here, three things are load-bearing, and each has a
test:

1. the tag must not appear in `index.html` in any form;
2. the click and scroll listeners must be created **inside** `start()`, after the
   yes — no listener at all is a different promise from a listener that stays
   quiet;
3. nothing may read the corpus. `track()` in `app.ts` sends counts, timings and
   preset names, and no analytics call may touch `#corpus`.

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
