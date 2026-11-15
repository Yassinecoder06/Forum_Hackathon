// Capture the real example images used by docs/diagrams from docs/diagrams/sample.json:
// source-video frames (ffmpeg) and avatar frames rendered by the running Studio at the
// same timestamps. Requires `make dev` (or equivalent) and CHROME_PATH or a Playwright Chromium.
import { chromium } from "playwright";
import { execFileSync } from "node:child_process";
import { mkdir, readFile, writeFile } from "node:fs/promises";
import path from "node:path";
import { fileURLToPath } from "node:url";

const root = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "..");
const sample = JSON.parse(await readFile(path.join(root, "docs/diagrams/sample.json"), "utf8"));
const out = path.join(root, "docs/diagrams/examples");
await mkdir(out, { recursive: true });

sample.times_ms.forEach((t, i) => {
  const crop = sample.video_crop ? `crop=${sample.video_crop},` : "";
  execFileSync("ffmpeg", ["-loglevel", "error", "-y", "-ss", String(t / 1000), "-i", path.join(root, sample.video),
    "-frames:v", "1", "-vf", `${crop}scale=480:-2`, path.join(out, `source_${i}.png`)]);
});

const browser = await chromium.launch({ headless: true,
  ...(process.env.CHROME_PATH ? { executablePath: process.env.CHROME_PATH } : {}),
  args: ["--enable-unsafe-swiftshader"] });
try {
  const page = await browser.newPage({ viewport: { width: 1500, height: 1100 }, deviceScaleFactor: 2 });
  await page.goto(process.env.APP_URL || "http://127.0.0.1:8080/studio/");
  await page.locator('[data-avatar-ready="true"]').waitFor({ timeout: 60000 });
  await page.getByLabel("Votre texte en français", { exact: true }).fill(sample.text);
  const composed = page.waitForResponse((r) => r.url().endsWith("/api/compose-animation") && r.request().method() === "POST");
  await page.getByRole("button", { name: "Créer l’animation", exact: true }).click();
  const plan = await (await composed).json();
  if (!plan.segments.length) throw new Error(`No sign found for "${sample.text}"`);
  // The avatar frames must come from the same recording as the source frames and skeleton.
  const used = plan.segments.map((s) => s.motion_id);
  if (sample.motion_id && (used.length !== 1 || used[0] !== sample.motion_id))
    throw new Error(`Studio played ${used.join(", ")} for "${sample.text}", not sample motion ${sample.motion_id}`);
  await page.getByRole("button", { name: "Mettre en pause", exact: true }).click();
  const slider = page.getByRole("slider", { name: "Position de l’animation", exact: true });
  const canvas = page.locator(".avatar-stage canvas");
  for (const [i, t] of sample.times_ms.entries()) {
    await slider.fill(String(t));
    await page.waitForTimeout(400);
    const data = await canvas.evaluate((el) => el.toDataURL("image/png"));
    await writeFile(path.join(out, `avatar_${i}.png`), Buffer.from(data.split(",")[1], "base64"));
  }
  console.log(`Captured ${sample.times_ms.length} source + avatar frames for ${sample.label} into docs/diagrams/examples/`);
} finally {
  await browser.close();
}
