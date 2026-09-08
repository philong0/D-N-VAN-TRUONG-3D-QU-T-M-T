// QA driver for 2026-08-24 Fix 1/2/3/4 — read-only browser check.
// Does NOT modify any source file or patient data — only navigates,
// screenshots, and reads console errors of the already-running dev server.
const { chromium } = require("playwright");

const PID = "257d9bfe-b246-4d82-a8c6-60a7ec076825";
const URL = `http://localhost:3000/patients/${PID}/studio`;
const OUT = "/home/ubuntu/dr-vantruong-3d-studio/scratchpad/qa_screenshots_20260824";

(async () => {
  const fs = require("fs");
  fs.mkdirSync(OUT, { recursive: true });

  const browser = await chromium.launch({ args: ["--no-sandbox"] });
  const page = await browser.newPage({ viewport: { width: 1600, height: 1000 } });
  const consoleErrors = [];
  page.on("console", (msg) => {
    if (msg.type() === "error") consoleErrors.push(msg.text());
  });
  page.on("pageerror", (err) => consoleErrors.push("PAGEERROR: " + err.message));

  console.log("Navigating to", URL);
  await page.goto(URL, { waitUntil: "domcontentloaded", timeout: 60000 });
  await page.waitForSelector("canvas", { timeout: 60000 });
  // Give the GNM multi-view fetch/geometry-build time to finish (cold ~few s once artifact is persisted).
  await page.waitForTimeout(6000);
  await page.screenshot({ path: `${OUT}/01_initial.png` });
  console.log("01_initial.png saved");

  // --- Rotation quick-view buttons ---
  const buttons = await page.locator("button", { hasText: /^-45°$|^0°$|^\+45°$/ }).allTextContents();
  console.log("Quick-view buttons found:", buttons);

  await page.locator("button", { hasText: "-45°" }).click();
  await page.waitForTimeout(1200);
  await page.screenshot({ path: `${OUT}/02_minus45.png` });

  await page.locator("button", { hasText: "0°" }).click();
  await page.waitForTimeout(1200);
  await page.screenshot({ path: `${OUT}/03_zero.png` });

  await page.locator("button", { hasText: "+45°" }).click();
  await page.waitForTimeout(1200);
  await page.screenshot({ path: `${OUT}/04_plus45.png` });

  // --- Manual drag beyond +45 — must clamp, never show back of head ---
  const canvasBox = await page.locator("canvas").first().boundingBox();
  const cx = canvasBox.x + canvasBox.width * 0.25; // left panel = "Trước"
  const cy = canvasBox.y + canvasBox.height * 0.5;
  await page.mouse.move(cx, cy);
  await page.mouse.down();
  await page.mouse.move(cx + 600, cy, { steps: 20 }); // large drag, should clamp
  await page.mouse.up();
  await page.waitForTimeout(500);
  await page.screenshot({ path: `${OUT}/05_drag_far_right.png` });

  await page.mouse.move(cx, cy);
  await page.mouse.down();
  await page.mouse.move(cx - 600, cy, { steps: 20 });
  await page.mouse.up();
  await page.waitForTimeout(500);
  await page.screenshot({ path: `${OUT}/06_drag_far_left.png` });

  // reset
  await page.locator("button", { hasText: "0°" }).click();
  await page.waitForTimeout(800);

  // --- Auto-rotate ---
  await page.locator("button", { hasText: "Tự xoay" }).click();
  await page.waitForTimeout(1500);
  await page.screenshot({ path: `${OUT}/07_autorotate_a.png` });
  await page.waitForTimeout(2500);
  await page.screenshot({ path: `${OUT}/08_autorotate_b.png` });
  await page.waitForTimeout(2500);
  await page.screenshot({ path: `${OUT}/09_autorotate_c.png` });
  await page.locator("button", { hasText: "Tự xoay" }).click(); // turn off
  await page.waitForTimeout(500);

  // --- Identity-only render mode ---
  await page.locator("button", { hasText: "Chỉ vùng mặt" }).click();
  await page.waitForTimeout(1500);
  await page.screenshot({ path: `${OUT}/10_identity_only.png` });
  await page.locator("button", { hasText: "45°" }).first().click().catch(() => {});
  await page.waitForTimeout(500);
  await page.screenshot({ path: `${OUT}/11_identity_only_45.png` });
  await page.locator("button", { hasText: "Chỉ vùng mặt" }).click(); // turn off
  await page.waitForTimeout(1000);
  await page.screenshot({ path: `${OUT}/12_full_again.png` });

  console.log("CONSOLE_ERRORS:", JSON.stringify(consoleErrors, null, 2));
  fs.writeFileSync(`${OUT}/console_errors.json`, JSON.stringify(consoleErrors, null, 2));

  await browser.close();
  console.log("DONE — screenshots in", OUT);
})().catch((err) => {
  console.error("QA SCRIPT FAILED:", err);
  process.exit(1);
});
