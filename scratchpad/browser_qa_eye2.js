const { chromium } = require('playwright');

const PID = process.argv[2] || '0913e4c9-999c-46cf-9d95-c294dff4bfd3';
const BASE = 'http://localhost:3000';
const OUT_DIR = '/home/ubuntu/dr-vantruong-3d-studio/scratchpad/qa_screenshots';

(async () => {
  const fs = require('fs');
  fs.mkdirSync(OUT_DIR, { recursive: true });
  const browser = await chromium.launch({ args: ['--no-sandbox', '--use-gl=swiftshader', '--enable-webgl', '--ignore-gpu-blocklist'] });
  const page = await browser.newPage({ viewport: { width: 1600, height: 1000 } });
  page.on('pageerror', (err) => console.log('pageerror:', err.message));

  await page.goto(`${BASE}/patients/${PID}/studio`, { waitUntil: 'domcontentloaded', timeout: 60000 });
  await page.waitForSelector('canvas', { timeout: 30000 });
  try { await page.waitForFunction(() => !document.body.innerText.includes('Đang tải'), { timeout: 60000 }); } catch {}
  await page.waitForTimeout(3000);
  await page.getByText('0°', { exact: true }).first().click().catch(() => {});
  await page.waitForTimeout(2000);

  // Expand "Mắt" once up front so layout is stable for BOTH shots (avoids the reflow-shift seen last run).
  const eyeSummary = page.locator('summary', { hasText: 'Mắt' }).first();
  await eyeSummary.scrollIntoViewIfNeeded();
  await eyeSummary.click({ timeout: 10000, force: true });
  await page.waitForTimeout(500);
  const slider = page.locator('#eye-creaseHeightMm');
  await slider.waitFor({ timeout: 10000 });

  // Locate the "SAU" (After) canvas specifically — SplitCompare renders two <canvas> elements, right one is "Sau".
  const canvases = page.locator('canvas');
  const count = await canvases.count();
  console.log('canvas count:', count);
  const afterCanvas = canvases.nth(count - 1); // right/last canvas = After panel

  await slider.fill('0');
  await slider.dispatchEvent('input');
  await slider.dispatchEvent('change');
  await page.waitForTimeout(1200);
  await afterCanvas.screenshot({ path: `${OUT_DIR}/eye2_crease0.png` });
  console.log('Saved eye2_crease0.png');

  await slider.fill('4');
  await slider.dispatchEvent('input');
  await slider.dispatchEvent('change');
  await page.waitForTimeout(1200);
  await afterCanvas.screenshot({ path: `${OUT_DIR}/eye2_crease4.png` });
  console.log('Saved eye2_crease4.png');

  await browser.close();
})().catch((err) => { console.error('QA script failed:', err); process.exit(1); });
