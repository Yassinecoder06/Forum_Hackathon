// Build the showcase's real example material from the running app (http://127.0.0.1:8080):
//  - site/data/examples.json: each featured phrase, its signs, durations, source-video URLs, credits
//  - site/media/<slug>.webm: the avatar signing it, recorded by the Studio's own WebM export button
// Usage: CHROME_PATH=... node e2e/build-showcase.mjs   (server must be running)
import { chromium } from "playwright";
import { mkdir, writeFile } from "node:fs/promises";
import path from "node:path";
import { fileURLToPath } from "node:url";
import { execFileSync } from "node:child_process";

const APP = process.env.APP_URL || "http://127.0.0.1:8080";
const root = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "..");
const site = path.join(root, "site");
await mkdir(path.join(site, "data"), { recursive: true });
await mkdir(path.join(site, "media"), { recursive: true });

const PHRASES = [
  "Je veux boire de l'eau.",
  "Je visite Paris avec ma famille.",
  "Bonjour, comment ça va ?",
  "Le musée est fermé demain.",
  "Je veux apprendre la langue des signes.",
];
const slug = (t) => t.normalize("NFD").replace(/[̀-ͯ]/g, "").toLowerCase().replace(/[^a-z0-9]+/g, "-").replace(/^-|-$/g, "");
const post = async (route, body) => (await fetch(`${APP}/api${route}`, { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body) })).json();

const library = (await (await fetch(`${APP}/api/library`)).json()).entries;
const byMotion = new Map(library.map((e) => [e.motion.motion_id, e]));
const examples = [];
for (const text of PHRASES) {
  const plan = await post("/compose-animation", { text });
  examples.push({
    text, slug: slug(text), duration_ms: plan.duration_ms, unknowns: plan.unknowns,
    video: `media/${slug(text)}.webm`,
    signs: plan.segments.map((s) => {
      const e = byMotion.get(s.motion_id);
      return { gloss: e.gloss.gloss, concept: e.gloss.concept_fr, duration_ms: s.duration_ms,
        source_video: `/api/captures/${e.capture_id}/source`, source_page: e.attribution?.source_url,
        credit: e.attribution?.credit, license: e.attribution?.license };
    }),
  });
}

const stats = await (await fetch(`${APP}/api/capabilities`)).json();


const browser = await chromium.launch({ headless: true,
  ...(process.env.CHROME_PATH ? { executablePath: process.env.CHROME_PATH } : {}),
  args: ["--no-sandbox","--enable-unsafe-swiftshader", "--autoplay-policy=no-user-gesture-required"] });
try {
  for (const ex of examples) {
    const page = await browser.newPage({ acceptDownloads: true, viewport: { width: 1500, height: 1000 } });
    const composed = page.waitForResponse((r) => r.url().endsWith("/api/compose-animation") && r.request().method() === "POST");
    await page.goto(`${APP}/studio/?text=${encodeURIComponent(ex.text)}`);
    await page.locator('[data-avatar-ready="true"]').waitFor({ timeout: 60000 });
    await composed;
    const exportButton = page.getByRole("button", { name: /Télécharger l’aperçu vidéo/ });
    await exportButton.waitFor({ state: "visible" });
    await page.waitForFunction(() => {
      const b = [...document.querySelectorAll("button")].find((x) => /Télécharger l’aperçu vidéo/.test(x.textContent || ""));
      return b && !b.disabled;
    }, null, { timeout: 60000 });
    const download = page.waitForEvent("download", { timeout: 120000 });
    await exportButton.click();
    await (await download).saveAs(path.join(site, ex.video));
    const videoPath = path.join(site, ex.video);
    ex.video_mp4 = ex.video.replace(/\.webm$/, ".mp4");
    ex.poster = ex.video.replace(/\.webm$/, ".jpg");
    execFileSync("ffmpeg", ["-y", "-i", videoPath, "-an", "-r", "30", "-c:v", "libx264", "-pix_fmt", "yuv420p", "-movflags", "+faststart", path.join(site, ex.video_mp4)], {stdio:"ignore"});
    execFileSync("ffmpeg", ["-y", "-ss", String(ex.duration_ms / 3000), "-i", videoPath, "-frames:v", "1", path.join(site, ex.poster)], {stdio:"ignore"});
    console.log(`Recorded ${ex.video} (${ex.signs.map((s) => s.gloss).join(" ")})`);
    await page.close();
  }
} finally {
  await browser.close();
}

// Publish metadata only after every genuine recording/transcode succeeds.
await writeFile(path.join(site, "data/examples.json"), JSON.stringify({
  generated: new Date().toISOString(), library_count: stats.library_count,
  dictasign: { videos: stats.dictasign?.total, imported: stats.dictasign?.imported, review: stats.dictasign?.quality_review_needed },
  examples,
}, null, 2));
