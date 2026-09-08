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
  page.on('console', (msg) => { if (msg.type() === 'error') consoleErrors.push(msg.text()); });
  page.on('pageerror', (err) => consoleErrors.push(`pageerror: ${err.message}`));

  const url = `${BASE}/patients/${PID}/studio`;
  console.log('Navigating to', url);
  await page.goto(url, { waitUntil: 'domcontentloaded', timeout: 60000 });
  await page.waitForSelector('canvas', { timeout: 30000 });
  try {
    await page.waitForFunction(() => !document.body.innerText.includes('Đang tải'), { timeout: 60000 });
  } catch { console.log('WARNING: loading text still present'); }
  await page.waitForTimeout(3000);

  await page.getByText('90°', { exact: true }).first().click().catch(() => {});
  await page.waitForTimeout(2000);
  await page.screenshot({ path: `${OUT_DIR}/eye_before_crease.png` });
  console.log('Saved eye_before_crease.png (creaseHeightMm=0, default)');

  // Expand the "Mắt" (eye) morph group and drag its crease-height slider to max
  // (summary's accessible text concatenates with its child ▾ span, so an
  // exact match against "Mắt" alone doesn't hit — match the <summary> element
  // that CONTAINS "Mắt" instead).
  const eyeSummary = page.locator('summary', { hasText: 'Mắt' }).first();
  await eyeSummary.scrollIntoViewIfNeeded();
  await eyeSummary.click({ timeout: 10000, force: true });
  console.log('Clicked "Mắt" group summary');
  await page.waitForTimeout(500);

  const slider = page.locator('#eye-creaseHeightMm');
  await slider.waitFor({ timeout: 10000 });
  await slider.fill('4'); // max per MORPH_SLIDERS.eye.sliders[0]: min 0, max 4, step 0.1
  await slider.dispatchEvent('input');
  await slider.dispatchEvent('change');
  console.log('Set creaseHeightMm slider to 4mm');
  await page.waitForTimeout(1500);
  await page.screenshot({ path: `${OUT_DIR}/eye_after_crease.png` });
  console.log('Saved eye_after_crease.png (creaseHeightMm=4)');

  console.log('\nConsole errors captured:', consoleErrors.length);
  for (const e of consoleErrors.slice(0, 30)) console.log(' -', e);

  await browser.close();
})().catch((err) => {
  console.error('QA script failed:', err);
  process.exit(1);
});
