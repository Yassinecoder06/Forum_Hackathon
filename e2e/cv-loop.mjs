import { chromium } from "playwright";
import assert from "node:assert/strict";
import { readFile, mkdir, writeFile } from "node:fs/promises";
const scenarios = JSON.parse(await readFile(new URL("../data/source_inventory/cv_phrases15_review.json", import.meta.url), "utf8"));
const phrases = scenarios.map(p => p.requested_concepts.join(" "));
const output = process.env.E2E_OUTPUT || "../artifacts/phrase-review/before";
await mkdir(output, { recursive: true });
const browser = await chromium.launch({ headless: true,
  ...(process.env.CHROME_PATH ? { executablePath: process.env.CHROME_PATH } : {}),
  args: ["--no-sandbox", "--enable-unsafe-swiftshader"] });
try {
  const page = await browser.newPage({ viewport: { width: 1500, height: 1100 }, acceptDownloads: true });
  const errors = [];
  let results = [];
  try { results = JSON.parse(await readFile(`${output}/results.json`, "utf8")); } catch {}
  page.on("pageerror", e => errors.push(e.message));
  await page.goto(process.env.APP_URL || "http://127.0.0.1:5173");
  await page.locator('[data-avatar-ready="true"]').waitFor({ timeout: 60000 });
  for (const [i, text] of phrases.entries()) {
    if (results.some(r => r.id === i + 1)) continue;
    const id = String(i + 1).padStart(2, "0");
    await page.getByLabel("Votre texte en français", { exact: true }).fill(text);
    const composed = page.waitForResponse(r => r.url().endsWith("/api/compose-animation") && r.request().method() === "POST");
    await page.getByRole("button", { name: "Créer l’animation", exact: true }).click();
    const response = await composed;
    assert.equal(response.status(), 200);
    const plan = await response.json();
    assert(plan.segments.length > 0);
    await page.getByRole("button", { name: "Mettre en pause", exact: true }).click();
    const lookup = await (await page.request.post("http://127.0.0.1:8000/gloss-lookup", { data: { text } })).json();
    const assets = await Promise.all(plan.segments.map(async s => [s.motion_id, await (await page.request.get("http://127.0.0.1:8000" + s.asset_url)).json()]));
    await writeFile(`${output}/${id}-input.json`, JSON.stringify({ text, plan, lookup, assets }, null, 2));
    const slider = page.getByRole("slider", { name: "Position de l’animation", exact: true });
    for (const fraction of [.2, .5, .8]) {
      const time = Math.round(plan.duration_ms * fraction / 10) * 10;
      await slider.fill(String(time)); await page.waitForTimeout(200);
      await page.locator(".avatar-stage").screenshot({ path: `${output}/${id}-${fraction}.png` });
    }
    const diagnostic = await page.locator('[data-rig-diagnostics]').evaluate(el => JSON.parse(el.dataset.rigDiagnostics));
    const download = page.waitForEvent("download", { timeout: Math.max(90000, plan.duration_ms + 30000) });
    await page.getByRole("button", { name: /Télécharger l’aperçu vidéo/ }).click();
    await (await download).saveAs(`${output}/${id}.webm`);
    assert((await readFile(`${output}/${id}.webm`)).length > 5000);
    await page.getByRole("button", { name: /Télécharger l’aperçu vidéo/ }).waitFor({ state: "visible" });
    await page.waitForFunction(() => !document.querySelector(".export-button")?.disabled);
    const result = { id: i + 1, text, source_phrase: scenarios[i].phrase, glosses: lookup.gloss_sequence.map(g => g.gloss), unknowns: plan.unknowns,
      duration_ms: plan.duration_ms, segments: plan.segments.length, diagnostic, video: `${id}.webm` };
    results.push(result);
    await writeFile(`${output}/results.json`, JSON.stringify(results, null, 2));
    console.log(JSON.stringify(result));
  }
  assert.deepEqual(errors, []);
} finally { await browser.close(); }
