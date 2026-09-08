import { chromium } from "playwright";
const browser = await chromium.launch({ args: ["--no-sandbox", "--disable-gpu"] });
const page = await browser.newPage({ viewport: { width: 900, height: 900 } });
page.setDefaultTimeout(15000);
await page.goto("http://localhost:3000/patients/0e9e1d90-2478-4d49-872f-c80772c4bc4f/studio", { waitUntil: "networkidle", timeout: 20000 });
await page.waitForTimeout(3000);
await page.screenshot({ path: "/tmp/claude-1001/-home-ubuntu-dr-vantruong-3d-studio/23269708-4f89-4f84-95c2-dca315fa6e91/scratchpad/probe.png" });
await browser.close();
console.log("DONE");
