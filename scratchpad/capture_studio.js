const { chromium } = require("playwright");

const PATIENT_ID = "0913e4c9-999c-46cf-9d95-c294dff4bfd3";
const OUT_DIR = "/tmp/claude-1001/-home-ubuntu-dr-vantruong-3d-studio/23269708-4f89-4f84-95c2-dca315fa6e91/scratchpad";

(async () => {
  const browser = await chromium.launch();
  const page = await browser.newPage({ viewport: { width: 1400, height: 900 } });
  page.on("console", (msg) => {
    if (msg.type() === "error" || msg.type() === "warning") {
      console.log(`[console.${msg.type()}] ${msg.text()}`);
    }
  });
  page.on("pageerror", (err) => console.log("[pageerror]", err.message));

  console.log("navigating...");
  await page.goto(`http://127.0.0.1:3000/patients/${PATIENT_ID}/studio`, { waitUntil: "domcontentloaded", timeout: 90000 });

  // Wait generously for the multi-view GNM fit fetch (known to take ~30-90s) to resolve
  // and the canvas to actually paint something (not just mount).
  await page.waitForSelector("canvas", { timeout: 20000 });
  console.log("canvas mounted, waiting for fit to resolve...");
  await page.waitForTimeout(90000);

  const views = [
    { label: "0°", file: "now_0deg.png" },
    { label: "45°", file: "now_45deg.png" },
    { label: "90°", file: "now_90deg.png" },
    { label: "Dưới lên", file: "now_below.png" },
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
  console.error("FATAL:", e);
  process.exit(1);
});
