import { chromium } from "playwright";

const browser = await chromium.launch({ args: ["--no-sandbox", "--disable-gpu", "--use-gl=swiftshader"] });
const page = await browser.newPage({ viewport: { width: 1400, height: 1000 }, deviceScaleFactor: 1 });
page.setDefaultTimeout(30000);

const url = "http://localhost:3000/patients/257d9bfe-b246-4d82-a8c6-60a7ec076825/studio";
console.log("Navigating to", url);
await page.goto(url, { waitUntil: "load", timeout: 30000 });
await page.waitForSelector("canvas", { timeout: 30000 });
await page.waitForTimeout(5000);

// Introspect the LIVE WebGL context actually running in the page
const glInfo = await page.evaluate(() => {
  const canvases = Array.from(document.querySelectorAll("canvas"));
  return canvases.map((c) => {
    const gl = c.getContext("webgl2") || c.getContext("webgl");
    if (!gl) return { hasGL: false, width: c.width, height: c.height, cssW: c.clientWidth, cssH: c.clientHeight };
    const attrs = gl.getContextAttributes();
    return {
      hasGL: true,
      width: c.width, height: c.height, cssW: c.clientWidth, cssH: c.clientHeight,
      devicePixelRatioUsed: c.width / c.clientWidth,
      contextAttributes: attrs,
      maxSamples: gl.getParameter(gl.SAMPLES !== undefined ? gl.SAMPLES : 0x8B4C),
    };
  });
});
console.log("window.devicePixelRatio:", await page.evaluate(() => window.devicePixelRatio));
console.log("Canvas/WebGL contexts found:", JSON.stringify(glInfo, null, 2));

await page.screenshot({ path: "/tmp/claude-1001/-home-ubuntu-dr-vantruong-3d-studio/60ab5e59-95c6-47f4-ae6e-c0263fe2ad59/scratchpad/bugE_full_0deg.png" });
console.log("Screenshot 0deg saved");

// Try clicking +90 / -90 buttons if present, for a profile view
const btn90 = page.getByText("+90°", { exact: true }).first();
if (await btn90.count() > 0) {
  await btn90.click();
  await page.waitForTimeout(1500);
  await page.screenshot({ path: "/tmp/claude-1001/-home-ubuntu-dr-vantruong-3d-studio/60ab5e59-95c6-47f4-ae6e-c0263fe2ad59/scratchpad/bugE_full_90deg.png" });
  console.log("Screenshot 90deg saved");
}

await browser.close();
console.log("DONE");
