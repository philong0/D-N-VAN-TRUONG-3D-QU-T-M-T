import { chromium } from "playwright";
import fs from "fs";

const OUT_DIR = "/tmp/claude-1001/-home-ubuntu-dr-vantruong-3d-studio/d4b5b860-02a9-4e13-bca5-402589d1b733/scratchpad/local_his";
fs.mkdirSync(OUT_DIR, { recursive: true });

(async () => {
  const browser = await chromium.launch();
  const page = await browser.newPage({ viewport: { width: 1440, height: 900 } });

  await page.goto("http://localhost:4000/", { waitUntil: "networkidle", timeout: 20000 });
  await page.waitForTimeout(1500);

  const bodyText = await page.evaluate(() => document.body.innerText).catch(() => "");
  fs.writeFileSync(`${OUT_DIR}/00_home.txt`, `URL: /\n\n${bodyText}`);

  const links = await page.evaluate(() => {
    return Array.from(document.querySelectorAll("a[href]")).map((a) => ({
      href: a.getAttribute("href"),
      text: a.textContent.trim().replace(/\s+/g, " "),
    })).filter((l) => l.text && l.href && l.href.startsWith("/") && l.href !== "/");
  });
  fs.writeFileSync(`${OUT_DIR}/links.json`, JSON.stringify(links, null, 2));
  console.log(`Found ${links.length} links`);

  const uniquePaths = [...new Set(links.map((l) => l.href))].slice(0, 15);
  for (const p of uniquePaths) {
    try {
      await page.goto(`http://localhost:4000${p}`, { waitUntil: "networkidle", timeout: 15000 });
      await page.waitForTimeout(1000);
      const t = await page.evaluate(() => document.body.innerText).catch(() => "");
      const name = p.replace(/\//g, "_") || "_root";
      fs.writeFileSync(`${OUT_DIR}${name}.txt`, `URL: ${p}\n\n${t}`);
      console.log(`OK ${p} (${t.length} chars)`);
    } catch (err) {
      console.log(`FAIL ${p}: ${err.message.slice(0, 100)}`);
    }
  }

  await browser.close();
})().catch((e) => { console.error("FATAL", e); process.exit(1); });
