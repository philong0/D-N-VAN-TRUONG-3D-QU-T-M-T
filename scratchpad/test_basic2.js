const { chromium } = require("playwright");
(async () => {
  const browser = await chromium.launch({ args: ["--no-sandbox"] });
  const page = await browser.newPage();
  console.log("navigating to 127.0.0.1:3000/patients...");
  await page.goto("http://127.0.0.1:3000/patients", { waitUntil: "domcontentloaded", timeout: 20000 });
  console.log("OK, title:", await page.title());
  await browser.close();
})().catch((e) => { console.error("FATAL:", e.message); process.exit(1); });
