const { chromium } = require('playwright');

const PID = process.argv[2] || '0913e4c9-999c-46cf-9d95-c294dff4bfd3';
const BASE = 'http://localhost:3000';
const OUT_DIR = '/home/ubuntu/dr-vantruong-3d-studio/scratchpad/qa_screenshots';

(async () => {
  const fs = require('fs');
  fs.mkdirSync(OUT_DIR, { recursive: true });

  const browser = await chromium.launch({ args: ['--no-sandbox', '--use-gl=swiftshader', '--enable-webgl', '--ignore-gpu-blocklist'] });
  const page = await browser.newPage({ viewport: { width: 1600, height: 1000 } });

  const consoleErrors = [];
  page.on('console', (msg) => {
    if (msg.type() === 'error') consoleErrors.push(msg.text());
  });
  page.on('pageerror', (err) => consoleErrors.push(`pageerror: ${err.message}`));

  const url = `${BASE}/patients/${PID}/studio`;
  console.log('Navigating to', url);
  await page.goto(url, { waitUntil: 'domcontentloaded', timeout: 60000 });

  // Wait for the WebGL canvas to appear
  await page.waitForSelector('canvas', { timeout: 30000 });
  console.log('Canvas found, waiting for render to settle...');
  await page.waitForTimeout(8000);
  // GNM head asset itself is ~53MB (fetched once, cached by the browser) —
  // "Đang tải GNM Head..." text means it's still in flight; poll for it to
  // disappear instead of a fixed guess.
  try {
    await page.waitForFunction(() => !document.body.innerText.includes('Đang tải'), { timeout: 60000 });
    console.log('Loading text gone, model should be ready');
  } catch {
    console.log('WARNING: still showing loading text after 60s extra wait');
  }
  await page.waitForTimeout(3000);

  await page.screenshot({ path: `${OUT_DIR}/00_initial.png` });
  console.log('Saved 00_initial.png');

  // Try clicking angle buttons by their Vietnamese label text seen in the UI
  const angleButtons = [
    { label: '0°', file: '01_angle_0.png' },
    { label: '45°', file: '02_angle_45.png' },
    { label: '90°', file: '03_angle_90.png' },
  ];

  for (const { label, file } of angleButtons) {
    try {
      const btn = page.getByText(label, { exact: true }).first();
      await btn.click({ timeout: 5000 });
      await page.waitForTimeout(2500); // camera transition
      await page.screenshot({ path: `${OUT_DIR}/${file}` });
      console.log(`Clicked "${label}", saved ${file}`);
    } catch (err) {
      console.log(`Could not click "${label}": ${err.message}`);
    }
  }

  console.log('\nConsole errors captured:', consoleErrors.length);
  for (const e of consoleErrors.slice(0, 30)) console.log(' -', e);

  await browser.close();
})().catch((err) => {
  console.error('QA script failed:', err);
  process.exit(1);
});
