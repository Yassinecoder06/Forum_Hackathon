import { chromium } from "playwright";
import assert from "node:assert/strict";
(async () => {
  const browser = await chromium.launch({
    ...(process.env.CHROME_PATH
      ? { executablePath: process.env.CHROME_PATH }
      : {}),
    headless: true,
    args: ["--no-sandbox", "--enable-unsafe-swiftshader"],
  });
  try {
    const page = await browser.newPage({
      viewport: { width: 1440, height: 1000 },
      acceptDownloads: true,
    });
    const errors = [];
    page.on("pageerror", (e) => errors.push(e.message));
    await page.goto(process.env.APP_URL || "http://127.0.0.1:5173");
    await page
      .getByRole("button", { name: "Espace local connecté", exact: true })
      .waitFor();
    await page
      .getByLabel("Votre texte en français", { exact: true })
      .fill("Demain, je vous dis bonjour.");
    await page.getByText("Préparer et corriger les concepts", {exact:true}).click();
    await page.getByLabel("Rechercher un concept").fill("bonjour");
    await page
      .locator(".concept-options")
      .getByRole("button", { name: "bonjour", exact: true })
      .click();
    await page.getByLabel("Rechercher un concept").fill("demain");
    await page
      .locator(".concept-options")
      .getByRole("button", { name: "demain", exact: true })
      .click();
    assert.equal(await page.locator(".concept-sequence li").count(), 2);
    await page.getByLabel("Monter le concept 2").click();
    assert.match(
      await page.locator(".concept-sequence li").first().innerText(),
      /demain/,
    );
    const ready = page.waitForResponse((r) =>
      r.url().endsWith("/compose-concepts"),
    );
    await page
      .getByRole("button", { name: "Prévisualiser cette sélection" })
      .click();
    const result = await (await ready).json();
    assert.equal(result.segments.length, 2);
    assert.equal(result.linguistic_review, "pending");
    await page.getByText("Sélection de concepts", { exact: true }).waitFor();
    const downloaded = page.waitForEvent("download");
    await page
      .getByRole("button", { name: "Télécharger la fiche de revue" })
      .click();
    await (await downloaded).saveAs("../artifacts/browser/concept-review.json");
    await page.screenshot({
      path: "../artifacts/browser/concepts.png",
      fullPage: true,
    });
    assert.equal(await page.getByRole("button", {name:"Proposer des concepts",exact:true}).count(),0);
    await page.setViewportSize({ width: 390, height: 844 });
    assert(
      await page.evaluate(
        () => document.documentElement.scrollWidth <= innerWidth + 1,
      ),
    );
    assert.deepEqual(errors, []);
    console.log(
      "Real editable concept sequence, ordering, preview, review export and mobile layout passed.",
    );
  } finally {
    await browser.close();
  }
})().catch((e) => {
  console.error(e);
  process.exit(1);
});
