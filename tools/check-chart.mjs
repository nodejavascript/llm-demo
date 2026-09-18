import { chromium } from 'playwright';

const URL = process.argv[2] ?? 'https://llm-demo.nodejavascript.com/';
const browser = await chromium.launch({ channel: 'chrome' });
const context = await browser.newContext({ viewport: { width: 1100, height: 900 }, deviceScaleFactor: 2 });
const page = await context.newPage();
const errors = [];
page.on('pageerror', (e) => errors.push('page: ' + e.message));
page.on('console', (m) => m.type() === 'error' && errors.push('console: ' + m.text()));

await page.goto(URL, { waitUntil: 'load' });
await page.waitForFunction(() => document.getElementById('trainBtn')?.textContent?.includes('parameters'), null, { timeout: 20000 });

const before = await page.evaluate(() => {
  const c = document.getElementById('lossChart');
  const box = c.getBoundingClientRect();
  return {
    exists: !!c,
    css: { w: box.width, h: box.height },
    backing: { w: c.width, h: c.height },
    visible: box.width > 0 && box.height > 0,
  };
});
console.log('before training:', JSON.stringify(before));
await page.locator('#step-3').screenshot({ path: '/tmp/chart-before.png' });

await page.click('#trainBtn');
await page.waitForTimeout(6000);
const during = await page.evaluate(() => {
  const c = document.getElementById('lossChart');
  const ctx = c.getContext('2d');
  const data = ctx.getImageData(0, 0, c.width, c.height).data;
  let painted = 0;
  for (let i = 3; i < data.length; i += 4) if (data[i] > 0) painted += 1;
  return {
    loss: document.getElementById('lossValue').textContent,
    backing: { w: c.width, h: c.height },
    paintedPixels: painted,
    of: (data.length / 4),
  };
});
console.log('during training:', JSON.stringify(during));
await page.locator('#step-3').screenshot({ path: '/tmp/chart-during.png' });

await page.waitForFunction(() => !document.getElementById('generateBtn').disabled, null, { timeout: 120000 });
await page.waitForTimeout(1500);
const after = await page.evaluate(() => {
  const c = document.getElementById('lossChart');
  const ctx = c.getContext('2d');
  const data = ctx.getImageData(0, 0, c.width, c.height).data;
  let painted = 0;
  for (let i = 3; i < data.length; i += 4) if (data[i] > 0) painted += 1;
  return { loss: document.getElementById('lossValue').textContent, paintedPixels: painted };
});
console.log('after training:', JSON.stringify(after));
await page.locator('#step-3').screenshot({ path: '/tmp/chart-after.png' });
console.log('errors:', errors.length ? errors : 'none');
await browser.close();
