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

/** A fresh page that records every request it makes and every analytics event. */
async function openPage() {
  const context = await browser.newContext();
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

  await page.goto(`${BASE}/`, { waitUntil: 'load' });
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

  assert.equal(await page.title(), 'Train a language model in your browser | llm-demo');
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
  assert.match(await page.locator('#statusMsg').textContent(), /Trained 600 steps in/);
  assert.match(await page.locator('#configSummary').textContent(), /600 steps · vocabulary 53/);

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
  const { context, page, requests } = await openPage();

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
  const { context, page, requests } = await openPage();
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
