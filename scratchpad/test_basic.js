const { chromium } = require("playwright");
(async () => {
  const browser = await chromium.launch({ args: ["--no-sandbox"] });
  const page = await browser.newPage();
  console.log("launched, navigating to example.com...");
  await page.goto("http://example.com", { waitUntil: "domcontentloaded", timeout: 15000 });
  console.log("example.com OK, title:", await page.title());
  console.log("navigating to localhost:3000/patients (list page)...");
  await page.goto("http://localhost:3000/patients", { waitUntil: "domcontentloaded", timeout: 20000 });
  console.log("patients list OK, title:", await page.title());
  await browser.close();
})().catch((e) => { console.error("FATAL:", e.message); process.exit(1); });
