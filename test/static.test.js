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
  const clientFiles = ['llm.js', 'trainer-host.js', 'trainer.worker.js', 'app.js', 'corpora.js'];
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

test('the only third-party request on the page is the analytics tag', () => {
  const urls = [...html.matchAll(/https?:\/\/[^"')\s]+/g)].map((m) => m[0]);
  // schema.org is a JSON-LD namespace identifier, not a request; the rest are
  // links a reader may follow, not resources the page loads.
  const allowedHosts = /googletagmanager\.com|google-analytics\.com|google\.com|github\.com|nodejavascript\.com|schema\.org/;
  const unexpected = urls.filter((u) => !allowedHosts.test(u));
  assert.deepEqual(unexpected, [], `unexpected external URL: ${unexpected.join(', ')}`);
  const loaded = [...html.matchAll(/<(?:script|link)[^>]*(?:src|href)="(https?:\/\/[^"]+)"/g)].map((m) => m[1]);
  const remote = loaded.filter((u) => !u.startsWith(CANONICAL) && !/googletagmanager\.com\/gtag/.test(u));
  assert.deepEqual(remote, [], `the page loads something remote: ${remote.join(', ')}`);
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
 * Analytics — the house standard for a new site
 * ------------------------------------------------------------------ */

test('the analytics tag is wired with the owner opt-out before it', () => {
  const id = html.match(/gtag\/js\?id=(G-[A-Z0-9]+)/);
  assert.ok(id, 'the tag is not wired');
  const measurementId = id[1];

  const optOutAt = html.indexOf('ga_opt_out');
  const tagAt = html.indexOf('googletagmanager.com/gtag/js');
  assert.ok(optOutAt > -1 && optOutAt < tagAt, 'the opt-out must run before the tag loads');
  assert.ok(html.includes(`ga-disable-${measurementId}`), 'the disable flag must name the real measurement id');
  assert.ok(html.includes(`?ga=off`) && html.includes(`?ga=on`), 'both opt-out and resume links must be offered');
  assert.ok(app.includes('scroll_depth'), 'scroll depth is part of the standard');
  assert.ok(app.includes('element_click'), 'the universal click event is part of the standard');
  assert.ok(app.includes('send_page_view: false'), 'page views are sent explicitly, never twice');
  assert.ok(app.includes('page_view'), 'a page view is sent');
});

test('the analytics id in the page is the one advertised to the script', () => {
  const id = html.match(/window\.LLM_DEMO_GA_ID\s*=\s*'([^']+)'/);
  assert.ok(id, 'the page must expose the measurement id to app.js');
  assert.ok(html.includes(`?id=${id[1]}`), 'the tag must use the same id');
  assert.ok(html.includes(`ga-disable-${id[1]}`), 'the opt-out must use the same id');
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
    'index.html', 'styles.css', 'app.js', 'llm.js', 'trainer-host.js', 'trainer.worker.js', 'corpora.js',
    'manifest.webmanifest', 'robots.txt', 'sitemap.xml', 'og.png',
    'favicon.ico', 'favicon.svg', 'favicon-32.png', 'apple-touch-icon.png',
    'android-chrome-192x192.png', 'android-chrome-512x512.png',
  ]);
  const present = readdirSync(SITE);
  const unexpected = present.filter((f) => !allowed.has(f));
  // site/ is published wholesale by rsync --delete: anything else would be served
  assert.deepEqual(unexpected, [], `site/ holds files that would be published: ${unexpected.join(', ')}`);
});
