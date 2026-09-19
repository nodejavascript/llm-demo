import { test, before, after } from 'node:test';
import assert from 'node:assert/strict';
import { spawn } from 'node:child_process';
import { fileURLToPath } from 'node:url';
import { dirname, join } from 'node:path';
import { chromium } from 'playwright';

/**
 * End-to-end tests: a real Chrome, the real page, the real worker, the real
 * download path — against `tools/serve.js` on a test port.
 *
 * The unit suite proves the model is arithmetic; this suite proves the *product*
 * works, and it is the only place two claims on the page can be checked at all:
 * that nothing the visitor types leaves the page, and that the assets really
 * arrive `no-store` (which is what replaced the version tokens — see app.ts).
 */

const here = dirname(fileURLToPath(import.meta.url));
const ROOT = join(here, '..');
const PORT = 4399;
const BASE = `http://127.0.0.1:${PORT}`;
const MARKER = 'ZZCORPUSMARKER4F2A';

let server;
let browser;

async function waitForServer(tries = 60) {
  for (let i = 0; i < tries; i++) {
    try {
      const response = await fetch(`${BASE}/`, { signal: AbortSignal.timeout(1000) });
      if (response.ok) return;
    } catch {
      /* not listening yet */
    }
    await new Promise((r) => setTimeout(r, 100));
  }
  throw new Error(`the local server never came up on ${BASE}`);
}

before(async () => {
  server = spawn(process.execPath, [join(ROOT, 'tools', 'serve.js'), String(PORT)], { stdio: 'ignore' });
  await waitForServer();
  browser = await chromium.launch({ channel: 'chrome' });
});

after(async () => {
  await browser?.close();
  server?.kill();
});

/**
 * A fresh page that records every request it makes and every analytics event.
 *
 * `consent` seeds the stored answer before any page script runs, which is how a
 * *returning* visitor is simulated — the choice made last week rather than one
 * made by clicking. `path` exists so the owner switch (`?ga=off`) can be tested.
 */
async function openPage(options = {}) {
  const { consent, path = '/' } = options;
  const context = await browser.newContext();
  if (consent) {
    await context.addInitScript((value) => {
      try {
        localStorage.setItem('analytics_consent', value);
      } catch (error) {
        /* storage blocked */
      }
    }, consent);
  }
  const page = await context.newPage();
  const requests = [];
  const pageErrors = [];
  const consoleErrors = [];

  page.on('request', (request) => requests.push(request));
  page.on('pageerror', (error) => pageErrors.push(String(error.message)));
  page.on('console', (message) => {
    if (message.type() === 'error') consoleErrors.push(message.text());
  });

  // The analytics beacons are answered here instead of being aborted: aborting
  // logs a console error, and this suite asserts there are none. Nothing leaves
  // the machine either way, and the requests are still recorded above.
  await page.route(/google-analytics\.com/, (route) => route.fulfill({ status: 204, body: '' }));

  await page.goto(`${BASE}${path}`, { waitUntil: 'load' });
  await page.waitForFunction(() => document.getElementById('trainBtn')?.textContent?.includes('parameters'));
  return { context, page, requests, pageErrors, consoleErrors };
}

const dataLayer = (page) => page.evaluate(() => (window.dataLayer ?? []).map((entry) => Array.from(entry)));
const eventNames = async (page) => (await dataLayer(page)).filter((e) => e[0] === 'event').map((e) => e[1]);

/* ------------------------------------------------------------------ *
 * Delivery
 * ------------------------------------------------------------------ */

test('every asset is served no-store, which is what keeps a deploy visible', async () => {
  const context = await browser.newContext();
  const request = context.request;
  for (const path of ['/', '/app.js', '/llm.js', '/trainer.worker.js', '/trainer-host.js', '/styles.css']) {
    const response = await request.get(BASE + path);
    assert.equal(response.status(), 200, `${path} did not answer 200`);
    const cacheControl = response.headers()['cache-control'] ?? '';
    assert.match(
      cacheControl,
      /no-store/,
      `${path} is cacheable (${cacheControl || 'no cache-control'}) — a deploy would be invisible to a returning visitor`
    );
  }
  await context.close();
});

test('the page loads with no errors and previews the model honestly', async () => {
  const { context, page, pageErrors, consoleErrors } = await openPage();

  // 🔴 Derived from the page's OWN canonical, not hard-coded and not from the URL.
  // This assertion used to pin the exact string "Train a language model in your
  // browser | llm-demo", so changing the title broke the browser suite — which is how
  // a test becomes a record of what the copy WAS rather than a check on the rule. The
  // served URL is no good either: the suite runs against 127.0.0.1, so its host is a
  // random port. The canonical is the site's declared identity, so the rule states
  // that the title IS that host — and it can never drift out of date again.
  const canonicalHost = await page.$eval('link[rel="canonical"]', (el) => new URL(el.href).host);
  assert.equal(
    await page.title(),
    canonicalHost,
    'the HTML title must be the full domain name, the host of the page’s canonical URL'
  );
  assert.equal(await page.locator('h1').count(), 1, 'exactly one h1');
  assert.match(await page.locator('#trainBtn').textContent(), /12,821 parameters/);

  const stats = await page.locator('#modelStats').innerText();
  assert.match(stats, /parameters/);
  assert.match(stats, /AdamW/);
  assert.match(await page.locator('#corpusStats').textContent(), /characters/);

  // The honesty block must actually name the comparison, not just say "small".
  assert.match(await page.locator('#paramAsides').textContent(), /GPT-2 small is 124 million/);

  assert.deepEqual(pageErrors, []);
  assert.deepEqual(consoleErrors, []);
  await context.close();
});

/* ------------------------------------------------------------------ *
 * The product
 * ------------------------------------------------------------------ */

test('trains a model, samples it, and offers the artefacts', async () => {
  const { context, page } = await openPage();

  await page.click('#trainBtn');
  await page.waitForFunction(() => document.getElementById('lossValue').textContent !== '—');
  await page.waitForFunction(() => !document.getElementById('generateBtn').disabled, null, { timeout: 180000 });
  await page.waitForFunction(() => document.querySelectorAll('#samples .sample').length >= 3, null, { timeout: 60000 });

  const loss = Number(await page.locator('#lossValue').textContent());
  assert.ok(Number.isFinite(loss) && loss > 0 && loss < 4, `loss looks wrong: ${loss}`);

  // The step count is READ from the page, not hard-coded here. It is decided by
  // the preset AND the corpus, through `plannedSteps` — and a literal in the test
  // is what broke everything when the presets gained the steps they needed. The
  // property worth asserting is not the number: it is that the run takes the
  // number it announced, and that the announcement is not a token amount.
  const statusText = await page.locator('#statusMsg').textContent();
  const summaryText = await page.locator('#configSummary').textContent();
  const announced = Number((summaryText.match(/([\d,]+)\s+steps/) ?? [])[1]?.replace(/,/g, ''));
  const trained = Number((statusText.match(/Trained ([\d,]+) steps in/) ?? [])[1]?.replace(/,/g, ''));
  assert.ok(announced > 0, `no step count announced in: ${summaryText}`);
  assert.equal(trained, announced, 'the run must take the number of steps it announced');
  assert.ok(trained >= 1000, `only ${trained} steps — the preset is being throttled`);
  assert.match(summaryText, /vocabulary 53/, 'the built-in name list was expected');

  const samples = await page.locator('#samples .sample').allTextContents();
  assert.equal(samples.length, 3);
  for (const sample of samples) {
    assert.ok(sample.trim().length > 0, 'a sample was empty');
    assert.ok(!/\uFFFD/.test(sample), 'a sample carried the unknown-vocabulary marker');
  }
  assert.ok((await page.locator('#vocabChars').textContent()).length > 20, 'the vocabulary was reported');

  // both downloads really fire, with the names the page promises
  const weights = page.waitForEvent('download');
  await page.click('#downloadWeights');
  assert.equal((await weights).suggestedFilename(), 'llm-demo-weights.json');

  const report = page.waitForEvent('download');
  await page.click('#downloadReport');
  assert.equal((await report).suggestedFilename(), 'llm-demo-training-report.md');

  await context.close();
});

test('stopping early still leaves a model that works', async () => {
  const { context, page } = await openPage();
  await page.click('#preset-standard');
  await page.click('#trainBtn');
  // let it take a few real steps, then stop
  const stepNow = () =>
    page.evaluate(() => Number((document.getElementById('stepValue')?.textContent ?? '0').split('/')[0].replace(/,/g, '').trim()));
  await page.waitForFunction(
    () => Number((document.getElementById('stepValue')?.textContent ?? '0').split('/')[0].replace(/,/g, '').trim()) > 8,
    null,
    { timeout: 60000 }
  );
  assert.ok((await stepNow()) > 8);
  await page.click('#stopBtn');
  await page.waitForFunction(() => document.querySelectorAll('#samples .sample').length >= 3, null, { timeout: 60000 });

  assert.match(await page.locator('#statusMsg').textContent(), /Stopped at \d+ steps/);
  const samples = await page.locator('#samples .sample').allTextContents();
  assert.equal(samples.length, 3);
  assert.equal(await page.locator('#downloadWeights').isEnabled(), true);
  await context.close();
});

test('sampling again honours the controls', async () => {
  const { context, page } = await openPage();
  await page.click('#trainBtn');
  await page.waitForFunction(() => !document.getElementById('generateBtn').disabled, null, { timeout: 180000 });
  await page.waitForFunction(() => document.querySelectorAll('#samples .sample').length >= 3, null, { timeout: 60000 });

  // top-k = 1 is the argmax, so every sample is the same string
  await page.locator('#topK').evaluate((el) => {
    const input = el;
    input.value = '1';
    input.dispatchEvent(new Event('input', { bubbles: true }));
  });
  assert.equal(await page.locator('#topKValue').textContent(), '1');
  await page.click('#generateBtn');
  await page.waitForFunction(() => document.getElementById('genMsg').textContent.startsWith('Sampled'));
  const samples = await page.locator('#samples .sample').allTextContents();
  assert.equal(samples.length, 3);
  assert.equal(new Set(samples).size, 1, 'top-k=1 must return the same greedy sample every time');

  // the shared seed is why: the three draws use seed + i*7919, so with top-k=1
  // they converge on identical text. A different temperature does not repeat.
  await context.close();
});

/* ------------------------------------------------------------------ *
 * The two claims the page makes
 * ------------------------------------------------------------------ */

test('nothing the visitor types leaves the page', async () => {
  // Consented on purpose: this is the worst case for the promise, because the
  // beacon path is live. A page nobody is counting is trivially silent.
  const { context, page, requests } = await openPage({ consent: 'granted' });

  // A corpus big enough to train on for the full run: a tiny one finishes in a
  // few steps and the Stop button is disabled again before a click can land.
  await page.locator('#corpus').fill(`${MARKER} one\n`.repeat(200));
  await page.locator('#corpus').dispatchEvent('input');
  await page.click('#trainBtn');
  await page.waitForFunction(() => document.getElementById('lossValue').textContent !== '—');
  await page.click('#stopBtn');
  await page.waitForFunction(() => document.querySelectorAll('#samples .sample').length >= 3, null, { timeout: 60000 });

  // every request the page made, and every analytics payload it queued
  const urls = requests.map((request) => request.url());
  const payloads = JSON.stringify(await dataLayer(page));

  for (const url of urls) {
    assert.ok(!url.includes(MARKER), `the corpus travelled to ${url}`);
    if (!/^http:\/\/127\.0\.0\.1:/.test(url)) {
      assert.match(
        url,
        /google-analytics\.com|googletagmanager\.com/,
        `the page talked to a host it should not: ${url}`
      );
    }
  }
  assert.ok(!payloads.includes(MARKER), 'the corpus reached the analytics payload');
  assert.ok(urls.some((u) => u.endsWith('/llm.js')) || urls.some((u) => u.endsWith('/app.js')), 'the modules loaded');
  assert.ok(
    !(await eventNames(page)).includes('model_trained'),
    'a stopped run is not a completed run and must not be recorded as one'
  );

  await context.close();
});

test('analytics carries the events the house standard asks for', async () => {
  const { context, page, requests } = await openPage({ consent: 'granted' });
  // A completed run, because model_trained is a conversion and only a finished
  // run earns it — a stopped run is not one, which the privacy test asserts.
  await page.click('#trainBtn');
  await page.waitForFunction(() => !document.getElementById('generateBtn').disabled, null, { timeout: 180000 });
  await page.waitForFunction(() => document.querySelectorAll('#samples .sample').length >= 3, null, { timeout: 60000 });
  await page.evaluate(() => window.scrollTo(0, document.body.scrollHeight));
  await page.waitForTimeout(300);

  const names = await eventNames(page);
  assert.ok(names.includes('page_view'), `no page_view (saw ${names.join(', ')})`);
  assert.ok(names.includes('train_started'), 'no train_started');
  assert.ok(names.includes('text_generated'), 'no text_generated');
  assert.ok(names.includes('scroll_depth'), 'no scroll_depth');
  assert.ok(names.includes('element_click'), 'no element_click');
  assert.ok(names.includes('model_trained'), 'a completed run must record model_trained');

  const trained = (await dataLayer(page)).find((e) => e[0] === 'event' && e[1] === 'model_trained');
  const params = trained[2] ?? {};
  assert.equal(params.preset, 'quick');
  assert.equal(params.parameters, 12821);

  // every beacon that did leave the browser went to analytics, nothing else
  const beacons = requests.map((r) => r.url()).filter((u) => !u.startsWith('http://127.0.0.1:'));
  assert.ok(beacons.length > 0, 'the analytics tag never fired');
  for (const url of beacons) {
    assert.match(url, /google-analytics\.com|googletagmanager\.com/);
  }
  await context.close();
});

/* ------------------------------------------------------------------ *
 * The cookie gate
 * ------------------------------------------------------------------ *
 *
 * The claim the whole site rests on is that the text you paste never leaves it.
 * The claim about the tag is narrower and harder: that a visit which refuses
 * makes NO request to Google at all. That is only true if Google's script is
 * absent rather than quiet, so these tests count real requests in a real browser
 * instead of trusting the code to behave.
 *
 * A note on what makes this worth testing at all: the previous version of this
 * page carried the tag in the head with an opt-out in front of it, and that is
 * not consent — the script was already loaded and the cookie already set before
 * the visitor was asked, so the "no" could only undo something that had already
 * happened. The difference is invisible in the page's source and obvious in the
 * network log, which is why the network log is where it is checked.
 */

const GOOGLE = /google-analytics\.com|googletagmanager\.com/;
const toGoogle = (requests) => requests.map((r) => r.url()).filter((u) => GOOGLE.test(u));

const barVisible = (page) => page.evaluate(() => !document.getElementById('consentBar').hidden);

async function settle(page) {
  await page.waitForFunction(() => document.getElementById('trainBtn')?.textContent?.includes('parameters'));
}

test('with no answer, nothing at all is asked of Google', async () => {
  const { context, page, requests } = await openPage();

  assert.deepEqual(toGoogle(requests), [], 'a visit that has not consented must contact nobody');
  assert.equal(await page.evaluate(() => typeof window.gtag), 'undefined', 'there must be no gtag to call');
  assert.equal(await page.evaluate(() => typeof window.llmTrack), 'undefined', 'the page must have no way to send an event');

  // The question is up; the panel behind it, and its switch, are not.
  assert.equal(await barVisible(page), true);
  assert.equal(await page.locator('#consentAsk').isVisible(), true);
  assert.equal(await page.locator('#consentPrefs').isVisible(), false);
  assert.equal(await page.locator('#consentAnalytics').isVisible(), false, 'the switch is one click behind the question');

  await context.close();
});

test('rejecting is a real answer, and it is remembered', async () => {
  const { context, page, requests } = await openPage();
  await page.click('#consentDecline');
  await page.waitForFunction(() => document.getElementById('consentBar').hidden);

  assert.deepEqual(toGoogle(requests), [], 'refusing must leave Google untouched');
  assert.equal(await page.evaluate(() => localStorage.getItem('analytics_consent')), 'denied');
  assert.equal(await page.evaluate(() => typeof window.gtag), 'undefined');

  // A refusal is not a nag: the next visit is not asked again.
  const again = await context.newPage();
  await again.goto(`${BASE}/`, { waitUntil: 'load' });
  await settle(again);
  assert.equal(await barVisible(again), false, 'the bar came back after a refusal');

  await context.close();
});

test('accepting loads the tag, and the model events only flow after that', async () => {
  const { context, page, requests } = await openPage();
  await page.click('#consentAccept');
  await page.waitForFunction(() => (window.dataLayer ?? []).some((entry) => entry[0] === 'event' && entry[1] === 'page_view'));

  const google = toGoogle(requests);
  assert.ok(
    google.some((url) => url.includes('googletagmanager.com/gtag/js')),
    `the tag must be fetched on a yes (saw ${google.join(', ')})`
  );
  assert.equal(await page.evaluate(() => typeof window.llmTrack), 'function', 'the page can send events once counted');
  assert.equal(await barVisible(page), false);

  // And the page's own events go through that gate.
  await page.click('#trainBtn');
  await page.waitForFunction(() =>
    (window.dataLayer ?? []).some((entry) => entry[0] === 'event' && entry[1] === 'train_started')
  );

  await context.close();
});

test('a remembered yes loads the tag without asking again', async () => {
  const { context, page, requests } = await openPage({ consent: 'granted' });
  assert.equal(await barVisible(page), false, 'a decision already made is not re-asked');
  assert.ok(toGoogle(requests).some((url) => url.includes('googletagmanager.com/gtag/js')));
  assert.equal(await page.evaluate(() => typeof window.llmTrack), 'function');
  await context.close();
});

test('the switch starts off, and moving it IS the answer', async () => {
  const { context, page } = await openPage();
  await page.click('#consentSettings');
  await page.waitForSelector('#consentPrefs:not([hidden])');

  // A box that arrives already ticked is a default, not a choice — and this is
  // the one place on the site where a default is the whole question.
  assert.equal(await page.locator('#consentAnalytics').getAttribute('aria-checked'), 'false');
  assert.equal(await page.locator('#consentAnalyticsWord').textContent(), 'Off');

  // No Save button: the switch is the answer, and the bar closes on it.
  await page.click('#consentAnalytics');
  await page.waitForFunction(() => document.getElementById('consentBar').hidden);
  assert.equal(await page.evaluate(() => localStorage.getItem('analytics_consent')), 'granted');

  await context.close();
});

test('the footer door opens the panel, and turning it off actually stops the counting', async () => {
  const { context, page, requests } = await openPage();
  await page.click('#consentAccept');
  await page.waitForFunction(() => typeof window.llmTrack === 'function');
  assert.ok(toGoogle(requests).length > 0, 'a consented visit is counted');

  // The footer door re-opens the panel rather than asking the question again, and
  // it leaves the answer alone — looking at a setting must not change it.
  await page.click('#consentBtn');
  await page.waitForSelector('#consentPrefs:not([hidden])');
  assert.equal(await page.locator('#consentAnalytics').getAttribute('aria-checked'), 'true');
  assert.equal(await page.locator('#consentAsk').isVisible(), false);

  // Withdrawing has to actually stop it. The tag is already in the page and
  // cannot be unloaded, so the honest version of "no" is a fresh page without it.
  requests.length = 0;
  await page.click('#consentAnalytics');
  await page.waitForFunction(() => typeof window.gtag === 'undefined', null, { timeout: 20000 });

  assert.deepEqual(toGoogle(requests), [], 'after withdrawing, Google is not contacted at all');
  assert.equal(await page.evaluate(() => localStorage.getItem('analytics_consent')), 'denied');
  assert.equal(await barVisible(page), false);

  await context.close();
});

test("the owner's own switch beats the visitor choice", async () => {
  // How George keeps his own visits out of his own numbers without changing what
  // every visitor is offered. It has to beat a stored "granted", or it is useless.
  const { context, page, requests } = await openPage({ consent: 'granted', path: '/?ga=off' });
  assert.deepEqual(toGoogle(requests), [], 'the owner switch must stop the tag loading');
  assert.equal(await page.evaluate(() => typeof window.gtag), 'undefined');
  assert.equal(await page.evaluate(() => localStorage.getItem('ga_opt_out')), '1');
  await context.close();
});

test('the cookie bar never covers the footer', async () => {
  // Found the hard way on inputresponse: a fixed banner takes the click, so a
  // footer link was unreachable until a visitor answered a question about
  // cookies. The page reserves the banner's own height instead of guessing it,
  // and the narrow width is the one that broke it there — the text re-wraps and
  // the banner settles taller than the space reserved for it.
  const { context, page } = await openPage();

  for (const width of [375, 1024]) {
    await page.setViewportSize({ width, height: 800 });
    await page.waitForTimeout(200);
    const state = await page.evaluate(() => {
      const door = document.getElementById('consentBtn');
      const bar = document.getElementById('consentBar');
      // The page scrolls smoothly, so a scroll and a measurement in the same tick
      // would measure where the element was, not where it is going.
      const wasSmooth = document.documentElement.style.scrollBehavior;
      document.documentElement.style.scrollBehavior = 'auto';
      door.scrollIntoView({ block: 'center' });
      document.documentElement.style.scrollBehavior = wasSmooth;
      const rect = door.getBoundingClientRect();
      const hit = document.elementFromPoint(rect.left + rect.width / 2, rect.top + rect.height / 2);
      return {
        reachable: hit === door || door.contains(hit),
        hit: hit ? hit.tagName + (hit.id ? `#${hit.id}` : '') : 'nothing',
        reserved: parseFloat(getComputedStyle(document.body).paddingBottom) || 0,
        barHeight: bar.hidden ? 0 : Math.ceil(bar.getBoundingClientRect().height),
      };
    });
    assert.ok(state.reachable, `at ${width}px the footer door is covered by ${state.hit}`);
    assert.ok(
      state.reserved >= state.barHeight,
      `at ${width}px the page reserves ${state.reserved}px but the bar is ${state.barHeight}px`
    );
  }

  // The click itself is the proof Playwright can give: it refuses to click an
  // element that something else is on top of.
  await page.click('#consentBtn');
  await page.waitForSelector('#consentPrefs:not([hidden])');

  await context.close();
});

/* ------------------------------------------------------------------ *
 * Layout, and the furniture a site is required to have
 * ------------------------------------------------------------------ */

test('no horizontal overflow at phone, tablet and desktop widths', async () => {
  const { context, page } = await openPage();
  for (const width of [320, 360, 414, 768, 1024, 1280, 1440]) {
    await page.setViewportSize({ width, height: 900 });
    await page.waitForTimeout(120);
    const overflow = await page.evaluate(() => {
      const de = document.documentElement;
      const culprits = [];
      document.querySelectorAll('body *').forEach((el) => {
        if (el.classList.contains('skip')) return; // parked off-canvas on purpose
        const box = el.getBoundingClientRect();
        if (box.width > 0 && box.right > de.clientWidth + 1) culprits.push(el.tagName.toLowerCase() + (el.id ? `#${el.id}` : ''));
      });
      return { pixels: de.scrollWidth - de.clientWidth, clientWidth: de.clientWidth, culprits: culprits.slice(0, 3) };
    });
    // A page measured in a panel with no viewport reports clientWidth 0 and an
    // "overflow" of the whole document — 233px here, twice mistaken for a layout
    // bug that did not exist. Refuse to measure rather than report a number.
    assert.ok(overflow.clientWidth > 200, `no usable viewport at ${width}px (clientWidth ${overflow.clientWidth})`);
    assert.equal(overflow.pixels, 0, `overflow of ${overflow.pixels}px at ${width}px (${overflow.culprits.join(', ')})`);
  }
  await context.close();
});

test('the icon set, manifest, robots and sitemap are all really served', async () => {
  const context = await browser.newContext();
  const request = context.request;
  const paths = [
    '/favicon.ico',
    '/favicon.svg',
    '/favicon-32.png',
    '/apple-touch-icon.png',
    '/android-chrome-192x192.png',
    '/android-chrome-512x512.png',
    '/og.png',
    '/manifest.webmanifest',
    '/robots.txt',
    '/sitemap.xml',
  ];
  for (const path of paths) {
    const response = await request.get(BASE + path);
    assert.equal(response.status(), 200, `${path} is referenced but does not answer 200`);
  }
  const manifest = await (await request.get(`${BASE}/manifest.webmanifest`)).json();
  assert.ok(manifest.icons.some((icon) => icon.sizes === '192x192'), 'Android needs a 192x192');
  const sitemap = await (await request.get(`${BASE}/sitemap.xml`)).text();
  assert.match(sitemap, /<loc>https:\/\/llm-demo\.nodejavascript\.com\/<\/loc>/);
  await context.close();
});
