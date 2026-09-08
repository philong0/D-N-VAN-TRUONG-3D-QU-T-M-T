const { chromium } = require("playwright");

const PATIENT_ID = "0913e4c9-999c-46cf-9d95-c294dff4bfd3";
const OUT_DIR = "/home/ubuntu/dr-vantruong-3d-studio/scratchpad";

(async () => {
  const browser = await chromium.launch({ args: ["--no-sandbox"] });
  const page = await browser.newPage({ viewport: { width: 1400, height: 900 } });

  page.on("console", (msg) => console.log(`[console.${msg.type()}] ${msg.text()}`));
  page.on("pageerror", (err) => console.log("[pageerror]", err.message));
  page.on("requestfailed", (req) => console.log("[requestfailed]", req.url(), req.failure()?.errorText));
  page.on("response", (res) => {
    if (!res.ok()) console.log("[response-not-ok]", res.status(), res.url());
  });

  console.log("navigating (waitUntil: commit)...");
  await page.goto(`http://127.0.0.1:3000/patients/${PATIENT_ID}/studio`, { waitUntil: "commit", timeout: 30000 });
  console.log("commit reached. Waiting for canvas element...");

  await page.waitForSelector("canvas", { timeout: 60000 });
  console.log("canvas mounted. Waiting for fit to resolve (up to 120s)...");
  await page.waitForTimeout(100000);

  const views = [
    { label: "0°", file: "live_0deg.png" },
    { label: "45°", file: "live_45deg.png" },
    { label: "90°", file: "live_90deg.png" },
    { label: "Dưới lên", file: "live_below.png" },
  ];

  for (const v of views) {
    const btn = page.getByRole("button", { name: v.label, exact: true });
    if (await btn.count() === 0) {
      console.log(`button not found: ${v.label}`);
      continue;
    }
    await btn.click();
    await page.waitForTimeout(3000);
    await page.screenshot({ path: `${OUT_DIR}/${v.file}` });
    console.log(`captured ${v.file}`);
  }

  await browser.close();
})().catch((e) => {
  console.error("FATAL:", e.message);
  process.exit(1);
});
