/**
 * Ad-hoc live check of the cookie gate. Not part of the suite — run by hand:
 *
 *   node tools/verify-live-consent.mjs
 *
 * The suite proves the gate against a local server. This proves it against the
 * deployed site, which is where the DNS, the TLS, the Caddy headers and the
 * cache purge all have to be right at the same time.
 */
import { chromium } from 'playwright';

const URL = 'https://llm-demo.nodejavascript.com/';
const GOOGLE = /google-analytics\.com|googletagmanager\.com/;

const browser = await chromium.launch({ channel: 'chrome' });
const results = [];

async function visit(label, { click, seed, query = '' } = {}) {
  const context = await browser.newContext();
  if (seed) {
    await context.addInitScript((value) => localStorage.setItem('analytics_consent', value), seed);
  }
  const page = await context.newPage();
  const google = [];
  page.on('request', (request) => {
    if (GOOGLE.test(request.url())) google.push(request.url());
  });
  page.on('pageerror', (error) => results.push(`  page error: ${error.message}`));
  await page.goto(URL + query, { waitUntil: 'load' });
  await page.waitForFunction(() => document.getElementById('trainBtn')?.textContent?.includes('parameters'));

  const before = google.length;
  if (click) {
    await page.click(click);
    await page.waitForTimeout(1200);
  }
  const state = await page.evaluate(() => ({
    gtag: typeof window.gtag,
    llmTrack: typeof window.llmTrack,
    bar: !document.getElementById('consentBar').hidden,
    reserved: parseFloat(getComputedStyle(document.body).paddingBottom) || 0,
    stored: localStorage.getItem('analytics_consent'),
  }));

  results.push(
    `${label}\n` +
      `  requests to Google before any click: ${before}\n` +
      `  requests to Google after:            ${google.length}\n` +
      `  window.gtag / window.llmTrack:       ${state.gtag} / ${state.llmTrack}\n` +
      `  bar visible / height reserved:       ${state.bar} / ${state.reserved}px\n` +
      `  stored answer:                       ${state.stored}`
  );
  await context.close();
  return { before, after: google.length, state };
}

const untouched = await visit('1. A visit that answers nothing');
const rejected = await visit('2. Reject all, clicked', { click: '#consentDecline' });
const accepted = await visit('3. Accept all, clicked', { click: '#consentAccept' });
const remembered = await visit('4. A remembered yes', { seed: 'granted' });
const owner = await visit('5. ?ga=off, the owner switch, against a remembered yes', {
  seed: 'granted',
  query: '?ga=off',
});

console.log('\n=== live cookie gate, ' + URL + ' ===\n');
console.log(results.join('\n\n'));

const checks = [
  ['nothing is asked of Google before a choice', untouched.before === 0],
  ['there is no gtag to call before a choice', untouched.state.gtag === 'undefined'],
  ['the page has no way to send an event before a choice', untouched.state.llmTrack === 'undefined'],
  ['the question is on screen', untouched.state.bar === true],
  ['refusing makes no request to Google at all', rejected.after === 0],
  ['refusing is remembered', rejected.state.stored === 'denied'],
  ['refusing closes the bar', rejected.state.bar === false],
  ['accepting fetches the tag', accepted.after > 0],
  ['accepting opens the page-events route', accepted.state.llmTrack === 'function'],
  ['a remembered yes loads the tag', remembered.after > 0],
  ['a remembered yes is not re-asked', remembered.state.bar === false],
  ['the owner switch beats a remembered yes', owner.after === 0 && owner.state.gtag === 'undefined'],
  ['the banner reserves its own height', untouched.state.reserved > 0],
];

console.log('\n=== checks ===');
let failed = 0;
for (const [name, ok] of checks) {
  if (!ok) failed += 1;
  console.log(`  ${ok ? 'PASS' : 'FAIL'}  ${name}`);
}
console.log(`\n${checks.length - failed}/${checks.length} passed`);

await browser.close();
process.exit(failed ? 1 : 0);
