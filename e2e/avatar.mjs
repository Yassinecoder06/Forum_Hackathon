import { chromium } from "playwright";
import assert from "node:assert/strict";
import { mkdir, writeFile } from "node:fs/promises";
const output = process.env.AVATAR_REVIEW_OUTPUT || "../artifacts/avatar-review/verified";
await mkdir(output, { recursive: true });
const browser = await chromium.launch({ headless: true,
  ...(process.env.CHROME_PATH ? { executablePath: process.env.CHROME_PATH } : {}),
  args: ["--no-sandbox", "--enable-unsafe-swiftshader"] });
try {
  const page = await browser.newPage({ viewport: { width: 1500, height: 1100 } });
  const errors = [];
  page.on("pageerror", (e) => errors.push(e.message));
  await page.goto(process.env.APP_URL || "http://127.0.0.1:5173");
  await page.locator('[data-avatar-ready="true"]').waitFor({ timeout: 60000 });
  const results = [];
  for (const text of ["Bonjour merci", "Bibliothèque", "Rencontrer ensemble laver"]) {
    await page.getByLabel("Votre texte en français", { exact: true }).fill(text);
    const composed = page.waitForResponse((r) => r.url().endsWith("/api/compose-animation") && r.request().method() === "POST");
    await page.getByRole("button", { name: "Créer l’animation", exact: true }).click();
    const response = await composed;
    assert.equal(response.status(), 200);
    const plan = await response.json();
    await page.getByRole("button", { name: "Mettre en pause", exact: true }).click();
    const slider = page.getByRole("slider", { name: "Position de l’animation", exact: true });
    const canvas = page.locator(".avatar-stage canvas");
    const first = Math.round(plan.duration_ms * .45 / 10) * 10;
    await slider.fill(String(first)); await page.waitForTimeout(250);
    const pixels = await canvas.evaluate((el) => el.toDataURL());
    await slider.fill("0"); await page.waitForTimeout(100);
    await slider.fill(String(first)); await page.waitForTimeout(250);
    assert.equal(await canvas.evaluate((el) => el.toDataURL()), pixels, "Actual rendered paused frame is deterministic after seeking");
    const diagnostics = await page.locator('[data-rig-diagnostics]').evaluate((el) => ({
      rig: JSON.parse(el.dataset.rigDiagnostics), hands: JSON.parse(el.dataset.handTracking),
    }));
    assert(diagnostics.rig.preparedMs > 0);
    assert(diagnostics.rig.maxCorrection <= .060001);
    await page.locator(".avatar-stage").screenshot({ path: `${output}/${results.length}-front.png` });
    const bounds = await canvas.boundingBox();
    await page.mouse.move(bounds.x + bounds.width * .5, bounds.y + bounds.height * .5);
    await page.mouse.down(); await page.mouse.move(bounds.x + bounds.width * .65, bounds.y + bounds.height * .5, { steps: 12 }); await page.mouse.up();
    await page.waitForTimeout(600);
    await page.locator(".avatar-stage").screenshot({ path: `${output}/${results.length}-side.png` });
    // Reset camera by reloading for a consistent next sample.
    results.push({ text, time_ms: first, ...diagnostics, pixel_seek_identical: true });
    if (results.length < 3) {
      await page.reload(); await page.locator('[data-avatar-ready="true"]').waitFor({ timeout: 60000 });
    }
  }
  assert.deepEqual(errors, []);
  await writeFile(`${output}/browser.json`, JSON.stringify(results, null, 2));
  console.log(JSON.stringify({ status: "passed", results }, null, 2));
} finally { await browser.close(); }
