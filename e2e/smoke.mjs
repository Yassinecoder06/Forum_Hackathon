import { chromium } from "playwright";
import assert from "node:assert/strict";
import { mkdir, readFile } from "node:fs/promises";

const base = process.env.APP_URL || "http://127.0.0.1:5173";
const output = process.env.E2E_OUTPUT || "../artifacts/browser";
await mkdir(output, { recursive: true });
const browser = await chromium.launch({
  headless: true,
  ...(process.env.CHROME_PATH
    ? { executablePath: process.env.CHROME_PATH }
    : {}),
  args: ["--no-sandbox", "--enable-unsafe-swiftshader"],
});
const context = await browser.newContext({
  viewport: { width: 1440, height: 1000 },
  acceptDownloads: true,
});
const page = await context.newPage();
page.setDefaultTimeout(20000);
const errors = [];
page.on("pageerror", (e) => errors.push(e.message));
let imported = null;
const api = async (path, options) => {
  const r = await fetch(`${base}/api${path}`, options);
  return r;
};
function wav() {
  const rate = 16000,
    count = rate * 3,
    b = Buffer.alloc(44 + count * 2);
  b.write("RIFF");
  b.writeUInt32LE(b.length - 8, 4);
  b.write("WAVEfmt ", 8);
  b.writeUInt32LE(16, 16);
  b.writeUInt16LE(1, 20);
  b.writeUInt16LE(1, 22);
  b.writeUInt32LE(rate, 24);
  b.writeUInt32LE(rate * 2, 28);
  b.writeUInt16LE(2, 32);
  b.writeUInt16LE(16, 34);
  b.write("data", 36);
  b.writeUInt32LE(count * 2, 40);
  for (let i = 0; i < count; i++)
    b.writeInt16LE(
      i >= rate && i < rate * 2
        ? 0
        : Math.round(Math.sin((2 * Math.PI * 220 * i) / rate) * 12000),
      44 + i * 2,
    );
  return b;
}
try {
  await page.goto(base);
  await page
    .getByRole("button", { name: "Espace local connecté", exact: true })
    .waitFor();
  await page
    .getByLabel("Votre texte en français", { exact: true })
    .fill("Bonjour, merci.");
  const composed = page.waitForResponse(
    (r) =>
      r.url().endsWith("/api/compose-animation") &&
      r.request().method() === "POST",
  );
  await page
    .getByRole("button", { name: "Créer l’animation", exact: true })
    .click();
  const plan = await (await composed).json();
  assert.equal(plan.segments.length, 2);
  const slider = page.getByRole("slider", {
    name: "Position de l’animation",
    exact: true,
  });
  await page.waitForFunction(
    (d) => document.querySelector("input[type=range]")?.max === String(d),
    plan.duration_ms,
  );
  await page
    .getByRole("button", { name: "Mettre en pause", exact: true })
    .click();
  await slider.fill("1500");
  assert.equal(await page.locator("canvas").count(), 1);
  await page.screenshot({ path: `${output}/studio.png`, fullPage: true });
  const download = page.waitForEvent("download");
  await page
    .getByRole("button", { name: /Télécharger l’aperçu vidéo/ })
    .click();
  const file = await download;
  assert.match(file.suggestedFilename(), /\.webm$/);
  await file.saveAs(`${output}/preview.webm`);
  assert(
    (await readFile(`${output}/preview.webm`)).length > 5000,
    "Video export is empty",
  );
  console.log("Composition, seeking and real WebM export passed.");

  await page
    .locator("input[type=file]")
    .first()
    .setInputFiles({
      name: "reference.wav",
      mimeType: "audio/wav",
      buffer: wav(),
    });
  await page
    .getByText("Hauteur moyenne", { exact: false })
    .waitFor({ timeout: 120000 });
  await page.screenshot({ path: `${output}/audio.png`, fullPage: true });
  const timed = page.waitForResponse(
    (r) =>
      r.url().endsWith("/api/compose-animation") &&
      r.request().method() === "POST",
  );
  await page
    .getByRole("button", { name: "Créer l’animation", exact: true })
    .click();
  const audioPlan = await (await timed).json();
  assert.equal(audioPlan.timing_mode, "audio_voiced");
  assert.equal(audioPlan.duration_ms, 3000);
  await page.waitForFunction(
    () => document.querySelector("input[type=range]")?.max === "3000",
  );
  console.log("Real WAV analysis and pause-aware timing passed.");

  await page.getByRole("button", { name: "Capturer", exact: true }).click();
  const fixture = JSON.parse(
    await readFile(
      new URL(
        "../data/approved_motion_assets/SYNTHETIC_001.json",
        import.meta.url,
      ),
      "utf8",
    ),
  );
  fixture.synthetic = true;
  await page
    .locator("input[type=file]")
    .setInputFiles({
      name: "test-motion.json",
      mimeType: "application/json",
      buffer: Buffer.from(JSON.stringify(fixture)),
    });
  await page.getByRole("checkbox", { name: /Je suis autorisé/ }).check();
  await page
    .getByRole("button", { name: "Extraire les mouvements", exact: true })
    .click();
  await page.getByText("Extraction terminée", { exact: true }).waitFor();
  const concept = `TestPipeline${Date.now()}`;
  await page.getByLabel("Concept français", { exact: true }).fill(concept);
  await page
    .getByLabel("Source de l’enregistrement", { exact: true })
    .fill("Automated original fixture; not a signer recording");
  await page
    .getByLabel("Droits et autorisation", { exact: true })
    .fill("CC0-1.0 technical test");
  const saved = page.waitForResponse(
    (r) => r.url().endsWith("/register") && r.request().method() === "POST",
  );
  await page
    .getByRole("button", { name: "Ajouter à la bibliothèque", exact: true })
    .click();
  const entry = await (await saved).json();
  imported = entry.motion.motion_id;
  assert.equal(entry.motion.synthetic, true);
  const lookup = await (
    await api("/gloss-lookup", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ text: concept }),
    })
  ).json();
  assert.equal(lookup.gloss_sequence[0].motion_id, imported);
  await page.getByRole("button", { name: "Bibliothèque", exact: true }).click();
  await page.getByText(concept, { exact: true }).waitFor();
  await page.screenshot({ path: `${output}/library.png`, fullPage: true });
  console.log("JSON capture, provenance, persistence and text lookup passed.");

  await page.getByRole("button", { name: "Studio", exact: true }).click();
  await page
    .getByLabel("Votre texte en français", { exact: true })
    .fill("Unmotentierementinconnu");
  await page
    .getByRole("button", { name: "Créer l’animation", exact: true })
    .click();
  await page.getByText("Aucun mouvement trouvé", { exact: true }).waitFor();
  await page.setViewportSize({ width: 390, height: 844 });
  await page.screenshot({ path: `${output}/mobile.png`, fullPage: true });
  assert(
    await page.evaluate(
      () => document.documentElement.scrollWidth <= innerWidth + 1,
    ),
    "Mobile overflow",
  );
  await page.route(
    (url) => url.pathname.startsWith("/api/"),
    (r) => r.abort(),
  );
  await page.reload();
  await page
    .getByRole("button", { name: "Service déconnecté", exact: true })
    .waitFor();
  await page
    .getByRole("button", { name: "Créer l’animation", exact: true })
    .click();
  await page.getByRole("alert").waitFor();
  assert.deepEqual(errors, []);
  console.log(
    "Unknown-word handling, mobile layout and offline errors passed.",
  );
} finally {
  if (imported) await api(`/library/${imported}`, { method: "DELETE" });
  await browser.close();
}
