import { chromium } from "playwright";
import assert from "node:assert/strict";
import { mkdir, readFile, writeFile } from "node:fs/promises";

const base = process.env.APP_URL || "http://127.0.0.1:5173";
const output = "../artifacts/dictasign";
await mkdir(output, { recursive: true });
const browser = await chromium.launch({
  headless: true,
  ...(process.env.CHROME_PATH ? { executablePath: process.env.CHROME_PATH } : {}),
  args: ["--no-sandbox", "--enable-unsafe-swiftshader"],
});
try {
  const page = await browser.newPage({ viewport: { width: 1440, height: 1080 }, acceptDownloads: true });
  const errors = [];
  page.on("pageerror", error => errors.push(error.message));
  await page.goto(base);
  await page.locator('[data-avatar-ready="true"]').waitFor({ timeout: 60000 });
  await page.getByLabel("Votre texte en français", { exact: true }).fill("Bonjour merci");
  const composition = page.waitForResponse(r => r.url().endsWith("/api/compose-animation") && r.request().method() === "POST");
  await page.getByRole("button", { name: "Créer l’animation", exact: true }).click();
  const plan = await (await composition).json();
  assert.equal(plan.segments.length, 2);
  const lookup = await (await page.request.post(base + "/api/gloss-lookup", { data: { text: "Bonjour merci" } })).json();
  assert.deepEqual(lookup.gloss_sequence.map(g => g.source_id), ["DICTASIGN_PORTAL_380", "DICTASIGN_PORTAL_905"]);
  await page.getByRole("button", { name: "Mettre en pause", exact: true }).click();
  await page.getByRole("slider", { name: "Position de l’animation", exact: true }).fill("600");
  await page.screenshot({ path: output + "/avatar.png", fullPage: true });
  const recording = page.waitForEvent("download", { timeout: 60000 });
  await page.getByRole("button", { name: /Télécharger l’aperçu vidéo/ }).click();
  await (await recording).saveAs(output + "/avatar.webm");
  assert((await readFile(output + "/avatar.webm")).length > 5000);
  await page.getByText("Exporter la chaîne glosses → poses", { exact: true }).click();
  const generated = page.waitForResponse(r => r.url().endsWith("/api/pipeline/lexical") && r.request().method() === "POST", { timeout: 280000 });
  await page.getByRole("button", { name: "Générer les poses et le diagnostic", exact: true }).click();
  const response = await generated;
  assert.equal(response.status(), 200);
  const result = await response.json();
  assert.deepEqual(result.glosses, ["BONJOUR", "MERCI"]);
  assert.equal(result.translation_validated, false);
  await page.locator(".pipeline-result video").waitFor();
  const report = await (await page.request.get(base + "/api" + result.report_url)).json();
  assert.deepEqual(report.sources.map(s => s.source_id), ["DICTASIGN_PORTAL_380", "DICTASIGN_PORTAL_905"]);
  assert.equal(Number(report.ffprobe.streams[0].nb_read_frames), result.frames);
  for (const [kind, name] of [["pose", "sequence.pose"], ["video", "skeleton.mp4"], ["gif", "skeleton.gif"]]) {
    const artifact = await page.request.get(base + "/api" + result[kind + "_url"]);
    assert.equal(artifact.status(), 200);
    await writeFile(output + "/" + name, await artifact.body());
  }
  await writeFile(output + "/result.json", JSON.stringify(report, null, 2));
  await page.screenshot({ path: output + "/pipeline.png", fullPage: true });
  const unknown = await page.request.post(base + "/api/pipeline/lexical", { data: { text: "Bonjour motinconnu" }, timeout: 280000 });
  assert.equal(unknown.status(), 422);
  assert.match((await unknown.json()).detail, /motinconnu/);
  assert.deepEqual(errors, []);
  console.log(JSON.stringify({ status: "passed", sources: report.sources, frames: result.frames, fps: result.fps, artifacts: output }, null, 2));
} finally {
  await browser.close();
}
