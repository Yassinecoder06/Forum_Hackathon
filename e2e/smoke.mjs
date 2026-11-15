import { chromium } from 'playwright';
import assert from 'node:assert/strict';
import { mkdir } from 'node:fs/promises';

const baseURL = process.env.APP_URL || 'http://127.0.0.1:5173';
const output = process.env.E2E_OUTPUT || '../artifacts/browser';
await mkdir(output, { recursive: true });
const browser = await chromium.launch({
  headless: true,
  ...(process.env.CHROME_PATH ? { executablePath: process.env.CHROME_PATH } : {}),
  args: ['--no-sandbox', '--enable-unsafe-swiftshader'],
});
const page = await browser.newPage({ viewport: { width: 1440, height: 1000 } });
const errors = [];
page.on('pageerror', error => errors.push(error.message));

function wav() {
  const rate = 16000, samples = rate * 3;
  const data = Buffer.alloc(44 + samples * 2);
  data.write('RIFF', 0); data.writeUInt32LE(data.length - 8, 4); data.write('WAVEfmt ', 8);
  data.writeUInt32LE(16, 16); data.writeUInt16LE(1, 20); data.writeUInt16LE(1, 22);
  data.writeUInt32LE(rate, 24); data.writeUInt32LE(rate * 2, 28);
  data.writeUInt16LE(2, 32); data.writeUInt16LE(16, 34); data.write('data', 36);
  data.writeUInt32LE(samples * 2, 40);
  for (let i = 0; i < samples; i++) {
    const value = i >= rate && i < rate * 2 ? 0 : Math.sin(2 * Math.PI * 220 * i / rate) * 12000;
    data.writeInt16LE(Math.round(value), 44 + i * 2);
  }
  return data;
}

try {
  await page.goto(baseURL);
  await page.getByText('API Port 8000 OK', { exact: true }).waitFor({ timeout: 15000 });
  const input = page.locator('input[type="text"]').first();
  await input.fill('bonjour merci');
  const composed = page.waitForResponse(r => r.url().endsWith('/api/compose-animation') && r.request().method() === 'POST');
  await page.getByRole('button', { name: 'Analyser & Composer', exact: true }).click();
  const response = await composed;
  assert.equal(response.status(), 200);
  const plan = await response.json();
  assert.equal(plan.segments.length, 2);
  assert.equal(plan.duration_ms, 2000);
  await page.waitForFunction(() => document.querySelector('input[aria-label="Position temporelle de l\'animation"]')?.max === '2000');
  assert.equal(await page.locator('canvas').count(), 1);
  await page.getByRole('button', { name: 'Mettre en pause', exact: true }).click();
  const timeline = page.getByRole('slider', { name: "Position temporelle de l'animation" });
  await timeline.fill('1500');
  await page.screenshot({ path: `${output}/studio.png`, fullPage: true });

  await page.getByRole('button', { name: 'Revue & Export', exact: true }).click();
  assert.equal(await page.locator('#reviewer-name-input').inputValue(), '');
  await page.locator('#reviewer-name-input').fill('automated_browser_test');
  await page.locator('#reviewer-notes-input').fill('Automated synthetic fixture test; no human linguistic validation.');
  await page.getByRole('button', { name: 'Rejeter', exact: true }).click();
  const reviewed = page.waitForResponse(r => r.url().endsWith('/api/render') && r.request().method() === 'POST');
  await page.getByRole('button', { name: 'Enregistrer la décision de revue (POST /review)', exact: true }).click();
  const reviewedPlan = await (await reviewed).json();
  assert.equal(reviewedPlan.plan.linguistic_review, 'rejected');
  assert.equal(reviewedPlan.video_exported, false);
  await page.getByText('Rejeté', { exact: true }).waitFor();
  assert((await page.locator('body').innerText()).includes(plan.plan_id));
  assert(!(await page.locator('body').innerText()).includes('undefined'));
  const blocked = page.waitForResponse(r => r.url().endsWith('/api/render') && r.request().method() === 'POST');
  await page.getByRole('button', { name: "Demander l'Export de Publication", exact: true }).click();
  assert.equal((await blocked).status(), 403);
  await page.getByRole('button', { name: 'Studio Texte-à-Signe', exact: true }).click();

  await input.fill('motinconnu');
  const unknown = page.waitForResponse(r => r.url().endsWith('/api/compose-animation') && r.request().method() === 'POST');
  await page.getByRole('button', { name: 'Analyser & Composer', exact: true }).click();
  const empty = await (await unknown).json();
  assert.equal(empty.segments.length, 0);
  assert.deepEqual(empty.unknowns, ['motinconnu']);
  await page.getByText('En attente', { exact: true }).waitFor();

  await page.getByRole('button', { name: 'Audio & Prosodie', exact: true }).click();
  const audioResponse = page.waitForResponse(r => r.url().endsWith('/api/analyze-audio'));
  await page.locator('input[type="file"]').setInputFiles({ name: 'tone-silence.wav', mimeType: 'audio/wav', buffer: wav() });
  const audio = await (await audioResponse).json();
  assert.equal(audio.duration_ms, 3000);
  assert(audio.frames.some(f => f.is_pause));
  assert(audio.frames.some(f => f.pitch_hz > 200 && f.pitch_hz < 240));
  assert.equal(audio.emotion_hypothesis, null);
  await page.getByText('Courbes Temporelles Réelles du Signal Audio', { exact: true }).waitFor();
  await page.screenshot({ path: `${output}/audio.png`, fullPage: true });

  await page.setViewportSize({ width: 390, height: 844 });
  await page.getByRole('button', { name: 'Studio Texte-à-Signe', exact: true }).click();
  await page.screenshot({ path: `${output}/mobile.png`, fullPage: true });
  assert(await page.evaluate(() => document.documentElement.scrollWidth <= window.innerWidth + 1), 'Mobile page overflows horizontally');

  await page.route(url => url.pathname.startsWith('/api/'), route => route.abort());
  await page.reload();
  await page.getByRole('button', { name: 'API Hors-ligne', exact: true }).waitFor();
  await page.getByRole('button', { name: 'Analyser & Composer', exact: true }).click();
  await page.getByText("Échec de l'appel API /api/gloss-lookup ou /api/compose-animation", { exact: true }).waitFor();
  assert(!await page.getByText('SYNTHETIC_001', { exact: true }).count(), 'Offline request silently loaded fixture');
  assert.deepEqual(errors, [], 'Browser raised runtime errors');
  console.log('Browser checks passed: two-clip playback, seeking, review persistence/publication gates, unknown-only clearing, WAV analysis, mobile layout, API-offline error.');
} finally {
  await browser.close();
}
