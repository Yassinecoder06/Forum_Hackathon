// Render every docs/diagrams/*.svg to a 2x PNG next to it (for slides and documents).
import { chromium } from "playwright";
import { readdir } from "node:fs/promises";
import path from "node:path";
import { fileURLToPath, pathToFileURL } from "node:url";

const dir = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "../docs/diagrams");
const browser = await chromium.launch({ headless: true,
  ...(process.env.CHROME_PATH ? { executablePath: process.env.CHROME_PATH } : {}) });
try {
  const page = await browser.newPage({ viewport: { width: 1600, height: 900 }, deviceScaleFactor: 2 });
  for (const name of (await readdir(dir)).filter((f) => f.endsWith(".svg")).sort()) {
    await page.goto(pathToFileURL(path.join(dir, name)).href);
    await page.waitForTimeout(150);
    await page.screenshot({ path: path.join(dir, name.replace(/\.svg$/, ".png")) });
  }
  console.log("PNG export complete");
} finally {
  await browser.close();
}
