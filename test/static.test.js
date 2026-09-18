import { test } from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync, existsSync, readdirSync } from 'node:fs';
import { fileURLToPath } from 'node:url';
import { dirname, join } from 'node:path';

const here = dirname(fileURLToPath(import.meta.url));
const SITE = join(here, '..', 'site');
const read = (name) => readFileSync(join(SITE, name), 'utf8');

const html = read('index.html');
const app = read('app.js');
const manifest = JSON.parse(read('manifest.webmanifest'));
const CANONICAL = 'https://llm-demo.nodejavascript.com/';

/* ------------------------------------------------------------------ *
 * Privacy: nothing the visitor types can leave the page
 * ------------------------------------------------------------------ */

test('no client-side file can send anything anywhere', () => {
  const clientFiles = ['llm.js', 'trainer-host.js', 'trainer.worker.js', 'app.js', 'corpora.js', 'consent.js'];
  const forbidden = [
    /\bfetch\s*\(/,
    /XMLHttpRequest/,
    /sendBeacon/,
    /new\s+WebSocket/,
    /navigator\.sendBeacon/,
    /EventSource/,
    /importScripts\s*\(/,
  ];
  for (const file of clientFiles) {
    const source = read(file);
    for (const pattern of forbidden) {
      assert.ok(!pattern.test(source), `${file} must not contain ${pattern}`);
    }
  }
});

test('the page loads NOTHING from another origin', () => {
  // This used to allow googletagmanager through, because the tag was in the head.
  // It is not any more: the consent gate appends it at runtime and only after a
  // yes, so a page that ships in this shape makes no third-party request at all
  // until a visitor chooses. If this test ever fails, something has been put back
  // into the page that a visitor cannot refuse.
  // A canonical link names this page's own address; it is not a load. Anything
  // with a remote src, or a remote stylesheet, would be.
  const loads = [...html.matchAll(/<(script|link)\b[^>]*>/g)].map((m) => m[0]).filter((tag) => !/rel="(canonical|alternate|preconnect|dns-prefetch)"/.test(tag));
  const loaded = loads
    .map((tag) => (tag.match(/(?:src|href)="(https?:\/\/[^"]+)"/) || [])[1])
    .filter(Boolean);
  assert.deepEqual(loaded, [], `the page loads something remote: ${loaded.join(', ')}`);
  assert.ok(!/googletagmanager|google-analytics/.test(html), 'no Google tag may sit in the page itself');
});

test('the textarea is never read into an analytics payload', () => {
  const valueReads = [...app.matchAll(/track\(([^;]*?)\);/gs)].map((m) => m[1]);
  for (const call of valueReads) {
    assert.ok(!/corpus'\)\.value|#corpus/.test(call), `an analytics call reads the corpus: ${call}`);
    assert.ok(!/\$\(\s*'corpus'\s*\)\.value/.test(call), `an analytics call reads the corpus: ${call}`);
  }
});

/* ------------------------------------------------------------------ *
 * The page and the script have to agree
 * ------------------------------------------------------------------ */

test('every element id the script looks up exists in the page', () => {
  const wanted = new Set();
  for (const m of app.matchAll(/\$\(\s*'([A-Za-z0-9_-]+)'\s*\)/g)) wanted.add(m[1]);
  // ids the script builds itself, and ids it reads dynamically
  const generated = new Set(['corpusButtons', 'presetButtons']);
  const dynamic = new Set(['temperatureValue', 'topKValue', 'lengthValue']);
  const missing = [...wanted].filter(
    (id) => !generated.has(id) && !dynamic.has(id) && !html.includes(`id="${id}"`)
  );
  assert.deepEqual(missing, [], `index.html is missing: ${missing.join(', ')}`);
});

test('every id the cookie gate looks up exists in the page', () => {
  // A typo here is the quietest failure the site could have: a misspelled id
  // means an answer button does nothing and the page says nothing about it. The
  // gate is small enough to check its ids exhaustively.
  const wanted = new Set([...read('consent.js').matchAll(/getElementById\('([A-Za-z0-9_-]+)'\)/g)].map((m) => m[1]));
  assert.ok(wanted.has('consentAccept') && wanted.has('consentDecline'), 'both answers must be wired');
  assert.ok(wanted.has('consentAnalytics'), 'the switch must be wired');
  assert.ok(wanted.has('consentBtn'), 'the footer door must be wired');
  const missing = [...wanted].filter((id) => !html.includes(`id="${id}"`));
  assert.deepEqual(missing, [], `index.html is missing: ${missing.join(', ')}`);
});

test('the script does not look up an id the page dropped', () => {
  const pageIds = new Set([...html.matchAll(/id="([A-Za-z0-9_-]+)"/g)].map((m) => m[1]));
  const idsBuiltByScript = new Set(['corpus-', 'preset-']);
  const looked = [...app.matchAll(/\$\(\s*'([A-Za-z0-9_-]+)'\s*\)/g)].map((m) => m[1]);
  for (const id of looked) {
    if (idsBuiltByScript.has(id)) continue;
    assert.ok(pageIds.has(id), `${id} is used by app.js but not in the page`);
  }
});

/* ------------------------------------------------------------------ *
 * SEO — the house standard
 * ------------------------------------------------------------------ */

test('titles and descriptions are inside the house limits', () => {
  const title = html.match(/<title>([^<]*)<\/title>/)[1];
  const description = html.match(/<meta\s+name="description"\s+content="([^"]*)"/)[1];
  assert.ok(title.length >= 15 && title.length <= 60, `title is ${title.length} characters: ${title}`);
  assert.ok(
    description.length >= 120 && description.length <= 160,
    `description is ${description.length} characters`
  );
});

test('canonical, social tags, one h1 and valid JSON-LD', () => {
  assert.ok(html.includes(`<link rel="canonical" href="${CANONICAL}"`), 'self-referencing canonical');
  for (const tag of ['og:title', 'og:description', 'og:image', 'og:url', 'og:type', 'twitter:card', 'twitter:image']) {
    assert.ok(html.includes(tag), `missing ${tag}`);
  }
  const h1s = [...html.matchAll(/<h1[\s>]/g)];
  assert.equal(h1s.length, 1, `expected exactly one <h1>, found ${h1s.length}`);

  const blocks = [...html.matchAll(/<script type="application\/ld\+json">([\s\S]*?)<\/script>/g)];
  assert.ok(blocks.length >= 1, 'a JSON-LD block');
  for (const block of blocks) {
    const parsed = JSON.parse(block[1].replace(/&quot;/g, '"'));
    assert.ok(parsed['@context'] === 'https://schema.org');
  }
});

test('every image on the page has alt text or is decorative', () => {
  for (const m of html.matchAll(/<img\b[^>]*>/g)) {
    assert.ok(/alt=/.test(m[0]) || /aria-hidden="true"/.test(m[0]), `image without alt: ${m[0]}`);
  }
});

/* ------------------------------------------------------------------ *
 * Icons — measured from the bytes, not assumed
 * ------------------------------------------------------------------ */

function pngSize(buffer) {
  assert.equal(buffer.readUInt32BE(0), 0x89504e47, 'not a PNG');
  return {
    width: buffer.readUInt32BE(16),
    height: buffer.readUInt32BE(20),
    bitDepth: buffer[24],
    colourType: buffer[25], // 6 = RGBA, 2 = RGB
  };
}

function icoSizes(buffer) {
  const count = buffer.readUInt16LE(4);
  const sizes = [];
  for (let i = 0; i < count; i++) {
    const offset = 6 + i * 16;
    sizes.push([buffer[offset] || 256, buffer[offset + 1] || 256]);
  }
  return sizes;
}

test('the icon set exists at the sizes the page declares', () => {
  const declared = [...html.matchAll(/<link rel="(?:icon|apple-touch-icon)"[^>]*href="\/([^"]+)"[^>]*?(?:sizes="([^"]+)")?/g)];
  assert.ok(declared.length >= 4, `expected at least four icon links, found ${declared.length}`);

  for (const [, full, sizes] of declared) {
    const path = join(SITE, full);
    assert.ok(existsSync(path), `${full} is linked but missing`);
    if (full.endsWith('.png') && sizes && /^\d+x\d+$/.test(sizes)) {
      const [w, h] = sizes.split('x').map(Number);
      const info = pngSize(readFileSync(path));
      assert.deepEqual([info.width, info.height], [w, h], `${full} declares ${sizes} but is ${info.width}x${info.height}`);
    }
  }
});

test('the apple touch icon has no alpha channel', () => {
  // iOS paints transparency black, so a transparent icon arrives on a home
  // screen with black wedges. This is the clause nodejavascript.com broke.
  const info = pngSize(readFileSync(join(SITE, 'apple-touch-icon.png')));
  assert.deepEqual([info.width, info.height], [180, 180], 'apple touch icon must be 180x180');
  assert.equal(info.colourType, 2, 'apple touch icon must be RGB with no alpha channel');
});

test('the ico carries the small sizes a browser asks for', () => {
  const sizes = icoSizes(readFileSync(join(SITE, 'favicon.ico')));
  for (const want of [16, 32, 48]) {
    assert.ok(sizes.some(([w]) => w === want), `favicon.ico has no ${want}x${want} entry (has ${sizes.map((s) => s.join('x')).join(', ')})`);
  }
});

test('the manifest and its icons all exist', () => {
  for (const icon of manifest.icons) {
    assert.ok(existsSync(join(SITE, icon.src.replace(/^\//, ''))), `manifest names a missing icon: ${icon.src}`);
  }
  assert.ok(html.includes('rel="manifest"'), 'the page must link the manifest');
  assert.ok(manifest.icons.some((i) => i.sizes === '192x192'), 'Android needs a 192x192');
});

/* ------------------------------------------------------------------ *
 * Discovery
 * ------------------------------------------------------------------ */

test('robots.txt points at the sitemap on the canonical host', () => {
  const robots = read('robots.txt');
  assert.match(robots, /^Sitemap:\s+https:\/\/llm-demo\.nodejavascript\.com\/sitemap\.xml$/m);
  assert.ok(!/Disallow:\s*\/\s*$/m.test(robots), 'the site must be crawlable');
});

test('the sitemap lists exactly the canonical URL and is well formed', () => {
  const sitemap = read('sitemap.xml');
  const locs = [...sitemap.matchAll(/<loc>([^<]+)<\/loc>/g)].map((m) => m[1]);
  assert.deepEqual(locs, [CANONICAL]);
  assert.ok(sitemap.trimStart().startsWith('<?xml version="1.0" encoding="UTF-8"?>'));
  assert.equal((sitemap.match(/<urlset/g) || []).length, 1);
  assert.equal((sitemap.match(/<\/url>/g) || []).length, locs.length);
  assert.match(sitemap, /<lastmod>\d{4}-\d{2}-\d{2}<\/lastmod>/);
});

/* ------------------------------------------------------------------ *
 * Analytics — behind the consent gate
 * ------------------------------------------------------------------ *
 *
 * The site used to carry the Google tag in the head with an opt-out in front of
 * it. That is not consent: the script is already loaded, the cookie is already
 * set, and the visitor is being asked to undo something that has happened. The
 * model now is inputresponse.com's — the tag is not in the page at all, and
 * consent.js is the only thing that may append it, only after a yes.
 *
 * These guards hold the shape of that in place. The behaviour is proved over
 * HTTP in test/e2e.test.js, which can count real requests to Google. */

const consent = read('consent.js');

test('consent.js is the only file allowed to name Google, and it loads the tag in one place', () => {
  assert.ok(!/googletagmanager/.test(app), 'app.js must not know how to load the tag');
  assert.ok(!/\bgtag\b/.test(app), 'app.js must send events through window.llmTrack, never gtag directly');

  const tagSources = [...consent.matchAll(/googletagmanager\.com\/gtag\/js/g)];
  assert.equal(tagSources.length, 1, 'the tag is appended from exactly one place');
  assert.match(consent, /createElement\(\s*['"]script['"]\s*\)/, 'the tag is built at runtime, not written into the page');
  assert.match(consent, /if \(started \|\| !allowed\(\)\)\s+return;/, 'nothing may append the tag before consent allows it');
  assert.match(consent, /function allowed\(\)/, 'consent is a named, readable predicate');
});

test('the measurement id rides on the consent script, never in the page', () => {
  const declared = html.match(/<script[^>]*src="\.\/consent\.js"[^>]*data-ga-id="(G-[A-Z0-9]+)"/);
  assert.ok(declared, 'the consent script must carry the measurement id as an attribute');
  assert.ok(!/G-[A-Z0-9]{6,}/.test(html.replace(declared[0], '')), 'the id must appear in the page only on that attribute');
  assert.ok(consent.includes('script[data-ga-id]'), 'consent.js reads the id from the attribute');
  assert.ok(consent.includes("OWNER_KEY = 'ga_opt_out'"), 'the owner opt-out key is unchanged');
  assert.ok(consent.includes("KEY = 'analytics_consent'"), 'the visitor choice is stored under the same key as inputresponse');
  assert.ok(consent.includes('?ga=off') || consent.includes('/?ga=off'), 'the owner can still leave himself out');
});

test('the events the house standard asks for live in the gate, not in the page', () => {
  for (const event of ['page_view', 'element_click', 'scroll_depth', 'send_page_view: false']) {
    assert.ok(consent.includes(event), `${event} must be sent from consent.js, where the gate is`);
  }
  assert.ok(consent.includes('25, 50, 75, 100'), 'scroll depth is marked at 25 / 50 / 75 / 100');
  // The listeners must be created inside start(), i.e. after the yes. A listener
  // that exists and stays quiet is not the same promise as no listener at all.
  // The first `'click'` in the file is the page-wide one, and it sits between
  // start() and build() — inside the gate, not in the page setup.
  const startAt = consent.indexOf('function start()');
  const clickAt = consent.indexOf("'click'");
  const buildAt = consent.indexOf('function build()');
  assert.ok(startAt > -1 && clickAt > startAt, 'the click listener is installed only once counting starts');
  assert.ok(clickAt < buildAt, 'and it is installed by the gate, not by the page setup');
});

test('the two answers are the same size, the same weight and the same class', () => {
  const accept = html.match(/<button[^>]*id="consentAccept"[^>]*>/);
  const decline = html.match(/<button[^>]*id="consentDecline"[^>]*>/);
  assert.ok(accept && decline, 'both answers must be offered');
  const cls = (tag) => (tag[0].match(/class="([^"]*)"/) || [])[1];
  assert.equal(cls(accept), cls(decline), 'the two answers must carry the same class');
  assert.equal(cls(accept), 'ghost', 'both answers wear the page\'s own secondary button, not the loud one');
});

test('the panel names the purpose and its switch ships off', () => {
  assert.match(html, /id="consentAnalytics"[^>]*aria-checked="false"/, 'the switch must arrive off: a pre-ticked box is a default, not a choice');
  assert.match(html, /Off unless you turn it on/, 'the panel says what the switch does');
  assert.match(html, /never anything you type and never the text\s+you train on/, 'the panel says what is never sent');
  assert.match(html, /href="#cookies"/, 'the ask links to the detail rather than carrying it');
  assert.match(html, /id="cookies"/, 'and that link must resolve to a real heading');
  assert.match(html, /id="consentDeviceRow"[^>]*hidden/, 'the owner row stays out of sight until it is used');
  assert.match(html, /id="consentBtn"/, 'the footer keeps one door to change the answer');
});

test('the banner cannot sit on top of the footer', () => {
  // Found the hard way on inputresponse: a fixed banner takes the click, so a
  // footer link was unreachable until a visitor answered a question about
  // cookies. The page reserves the banner's own height, measured at runtime.
  const css = read('styles.css');
  assert.match(css, /padding-bottom:\s*var\(--consent-height/, 'the page must reserve the banner height');
  assert.match(consent, /setProperty\('--consent-height'/, 'consent.js must measure it, not guess');
  assert.match(consent, /ResizeObserver/, 'and keep following it when the text re-wraps');
  assert.match(css, /\[hidden\]\s*\{[^}]*display:\s*none\s*!important/, '`hidden` must beat the display rules the panel switches between');
});

test('no asset reference carries a version query', () => {
  // The first fix for a stale-module bug was a `?v=` token on every reference.
  // That cannot survive TypeScript, which cannot resolve an import specifier with
  // a query on it — so the mechanism is now `Cache-Control: no-store` on the whole
  // site, set in the Caddy block. This test keeps the two honest: a query string
  // creeping back in means somebody has started doing it two ways again, and the
  // header that actually guarantees freshness is asserted over HTTP by the
  // end-to-end suite (test/e2e.test.js).
  const files = ['index.html', 'app.js', 'trainer-host.js', 'trainer.worker.js'];
  for (const file of files) {
    const source = read(file);
    for (const match of source.matchAll(/\.\/[A-Za-z0-9_.-]+\.(?:js|css)\?[A-Za-z0-9=._-]*/g)) {
      assert.fail(`${file} carries a version query again: ${match[0]}`);
    }
  }
});

/* ------------------------------------------------------------------ *
 * The site is complete
 * ------------------------------------------------------------------ */

test('every control that starts disabled is enabled again by the script', () => {
  // A button that ships disabled and is never enabled is a dead control, and it
  // is invisible on a quick read of the page. This is how the two download
  // buttons were found never being enabled on 2026-09-17.
  const disabledIds = [
    ...html.matchAll(/<(?:button|input|select|textarea)[^>]*?id="([A-Za-z0-9_-]+)"[^>]*?\bdisabled\b[^>]*>/g),
  ].map((m) => m[1]);
  assert.ok(disabledIds.length >= 3, `expected several controls to start disabled, found ${disabledIds.length}`);

  const enabled = new Set();
  // $('id').disabled = false  ·  $<HTMLButtonElement>('id').disabled = false
  for (const m of app.matchAll(/\$[A-Za-z]*\s*(?:<[^>]*>)?\s*\(\s*'([A-Za-z0-9_-]+)'\s*\)\.disabled\s*=\s*false/g)) {
    enabled.add(m[1]);
  }
  // for (const id of ['a', 'b']) { $<HTMLButtonElement>(id).disabled = false; }
  for (const m of app.matchAll(/for\s*\(\s*const\s+id\s+of\s*\[([^\]]+)\]\s*\)[^}]*?\.disabled\s*=\s*false/gs)) {
    for (const id of m[1].matchAll(/'([A-Za-z0-9_-]+)'/g)) enabled.add(id[1]);
  }

  for (const id of disabledIds) {
    assert.ok(enabled.has(id), `${id} starts disabled and nothing ever enables it`);
  }
});

test('nothing in site/ is a stray working file', () => {
  const allowed = new Set([
    'index.html', 'styles.css', 'app.js', 'consent.js', 'llm.js', 'trainer-host.js', 'trainer.worker.js', 'corpora.js',
    'manifest.webmanifest', 'robots.txt', 'sitemap.xml', 'og.png',
    'favicon.ico', 'favicon.svg', 'favicon-32.png', 'apple-touch-icon.png',
    'android-chrome-192x192.png', 'android-chrome-512x512.png',
  ]);
  const present = readdirSync(SITE);
  const unexpected = present.filter((f) => !allowed.has(f));
  // site/ is published wholesale by rsync --delete: anything else would be served
  assert.deepEqual(unexpected, [], `site/ holds files that would be published: ${unexpected.join(', ')}`);
});
