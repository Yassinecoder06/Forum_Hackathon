/**
 * Guided demo tour (~90 s). Loaded only with `?demo` in the URL (see main.tsx); it drives the real UI
 * (real engines, real models, nothing pre-rendered) with an on-screen cursor and captions, for screen recording.
 *
 * Media come from /demo-media/ (frontend/public/demo-media/, git-ignored: cut locally from the private dataset).
 * Flow: load -> models warm up in the background -> "press Enter" card -> start recording -> Enter -> tour.
 *
 * `?demo&fast` skips the waits between scenes (dry runs). `?demo&from=film` starts at a scene.
 */

const params = new URLSearchParams(location.search);
const FAST = params.has("fast");
const FROM = params.get("from");
const MEDIA = "/demo-media";

const sleep = (ms: number) => new Promise((r) => setTimeout(r, FAST ? Math.min(ms, 150) : ms));
const realSleep = (ms: number) => new Promise((r) => setTimeout(r, ms));

// ------------------------------------------------------------------ overlay
const css = `
.demo-cursor{position:fixed;left:50%;top:60%;width:22px;height:22px;margin:-11px 0 0 -11px;border-radius:50%;
  background:rgba(255,255,255,.95);border:3px solid #5a2f1c;box-shadow:0 4px 14px rgba(0,0,0,.35);z-index:2147483646;
  pointer-events:none;transition:left .5s cubic-bezier(.4,0,.2,1),top .5s cubic-bezier(.4,0,.2,1),transform .15s}
.demo-cursor.down{transform:scale(.75)}
.demo-ripple{position:fixed;width:16px;height:16px;margin:-8px 0 0 -8px;border-radius:50%;border:3px solid #c0703f;
  z-index:2147483645;pointer-events:none;animation:demo-ripple .6s ease-out forwards}
@keyframes demo-ripple{to{transform:scale(4.5);opacity:0}}
.demo-ring{position:fixed;border:3px solid #c0703f;border-radius:14px;z-index:2147483644;pointer-events:none;
  box-shadow:0 0 0 6px rgba(192,112,63,.18);transition:all .45s ease;opacity:0}
.demo-caption{position:fixed;left:50%;bottom:34px;transform:translate(-50%,20px);max-width:min(880px,86vw);
  background:rgba(40,22,14,.92);color:#fff;border-radius:18px;padding:16px 26px 17px;z-index:2147483647;
  font-family:'DM Sans',system-ui,sans-serif;box-shadow:0 12px 40px rgba(0,0,0,.35);opacity:0;
  transition:opacity .45s ease,transform .45s ease;pointer-events:none}
.demo-caption.on{opacity:1;transform:translate(-50%,0)}
.demo-caption b{display:block;font-size:24px;font-weight:700;letter-spacing:.2px}
.demo-caption span{display:block;font-size:16px;opacity:.86;margin-top:4px}
.demo-caption i{font-style:normal;font-size:12px;letter-spacing:2px;color:#e7a77c;display:block;margin-bottom:5px}
.demo-gate{position:fixed;inset:0;display:grid;place-items:center;background:rgba(30,16,10,.55);z-index:2147483647;
  font-family:'DM Sans',system-ui,sans-serif}
.demo-gate div{background:#fff;border-radius:20px;padding:28px 36px;text-align:center;box-shadow:0 20px 60px rgba(0,0,0,.4)}
.demo-gate b{display:block;font-size:22px;color:#3a1f12}
.demo-gate small{display:block;margin-top:8px;color:#7a5a48;font-size:14px}
.demo-title{position:fixed;inset:0;display:grid;place-items:center;z-index:2147483646;pointer-events:none;
  background:radial-gradient(circle at 50% 45%,rgba(40,22,14,.78),rgba(25,13,8,.92));color:#fff;opacity:0;
  transition:opacity .8s ease;font-family:'Playfair Display',Georgia,serif;text-align:center}
.demo-title.on{opacity:1}
.demo-title h1{font-size:64px;margin:0;font-weight:600;color:#fff}
.demo-title h1 em{color:#e7a77c}
.demo-title p{font-family:'DM Sans',system-ui,sans-serif;font-size:22px;color:#fff;opacity:.88;margin-top:14px}
`;

let cursor: HTMLDivElement, ring: HTMLDivElement, caption: HTMLDivElement;

function overlay() {
  const style = document.createElement("style");
  style.textContent = css;
  document.head.append(style);
  cursor = Object.assign(document.createElement("div"), { className: "demo-cursor" });
  ring = Object.assign(document.createElement("div"), { className: "demo-ring" });
  caption = Object.assign(document.createElement("div"), { className: "demo-caption" });
  document.body.append(ring, cursor, caption);
}

async function say(step: string, title: string, sub = "") {
  caption.classList.remove("on");
  await realSleep(FAST ? 50 : 350);
  caption.innerHTML = `${step ? `<i>${step}</i>` : ""}<b>${title}</b>${sub ? `<span>${sub}</span>` : ""}`;
  caption.classList.add("on");
}
const hush = () => caption.classList.remove("on");

async function titleCard(h: string, p: string, ms: number) {
  const card = Object.assign(document.createElement("div"), { className: "demo-title" });
  card.innerHTML = `<div><h1>${h}</h1><p>${p}</p></div>`;
  document.body.append(card);
  await realSleep(30);
  card.classList.add("on");
  await sleep(ms);
  card.classList.remove("on");
  await realSleep(FAST ? 50 : 800);
  card.remove();
}

function highlight(el: Element | null) {
  if (!el) {
    ring.style.opacity = "0";
    return;
  }
  const r = el.getBoundingClientRect();
  Object.assign(ring.style, { left: `${r.left - 6}px`, top: `${r.top - 6}px`, width: `${r.width + 12}px`, height: `${r.height + 12}px`, opacity: "1" });
}

async function moveTo(el: Element) {
  el.scrollIntoView({ behavior: FAST ? "auto" : "smooth", block: "center" });
  await sleep(380);
  const r = el.getBoundingClientRect();
  cursor.style.left = `${r.left + Math.min(r.width / 2, 60)}px`;
  cursor.style.top = `${r.top + r.height / 2}px`;
  await sleep(520);
}

async function click(el: Element | null, pause = 350) {
  if (!el) throw new Error("demo: missing element");
  await moveTo(el);
  cursor.classList.add("down");
  const r = cursor.getBoundingClientRect();
  const ripple = Object.assign(document.createElement("div"), { className: "demo-ripple" });
  Object.assign(ripple.style, { left: `${r.left + r.width / 2}px`, top: `${r.top + r.height / 2}px` });
  document.body.append(ripple);
  setTimeout(() => ripple.remove(), 700);
  await realSleep(120);
  cursor.classList.remove("down");
  (el as HTMLElement).click();
  await sleep(pause);
}

// ------------------------------------------------------------------ DOM helpers
const $ = <T extends Element = HTMLElement>(sel: string, root: ParentNode = document) => root.querySelector<T>(sel);
const byText = (sel: string, text: string | RegExp) =>
  [...document.querySelectorAll<HTMLElement>(sel)].find((e) => (typeof text === "string" ? e.innerText.trim().startsWith(text) : text.test(e.innerText)));

async function waitFor<T>(fn: () => T | null | undefined | false, timeout = 30000, step = 120): Promise<T> {
  const end = performance.now() + timeout;
  for (;;) {
    const v = fn();
    if (v) return v;
    if (performance.now() > end) throw new Error("demo: timeout");
    await realSleep(step);
  }
}

function setNative(el: HTMLInputElement | HTMLTextAreaElement | HTMLSelectElement, value: string) {
  const proto = el instanceof HTMLTextAreaElement ? HTMLTextAreaElement.prototype : el instanceof HTMLSelectElement ? HTMLSelectElement.prototype : HTMLInputElement.prototype;
  Object.getOwnPropertyDescriptor(proto, "value")!.set!.call(el, value);
  el.dispatchEvent(new Event(el instanceof HTMLSelectElement ? "change" : "input", { bubbles: true }));
}

async function typeInto(el: HTMLTextAreaElement, text: string) {
  await click(el, 150);
  setNative(el, "");
  for (let i = 1; i <= text.length; i++) {
    setNative(el, text.slice(0, i));
    await realSleep(FAST ? 0 : 28 + Math.random() * 30);
  }
  await sleep(300);
}

async function attach(input: HTMLInputElement, url: string, name: string, type: string) {
  const blob = await (await fetch(url)).blob();
  const dt = new DataTransfer();
  dt.items.add(new File([blob], name, { type }));
  input.files = dt.files;
  input.dispatchEvent(new Event("change", { bubbles: true }));
}

const textArea = () => $<HTMLTextAreaElement>("textarea")!;
const enginePanel = () => $<HTMLDetailsElement>("details.lsf-engine-panel")!;
const motionInfo = () => $(".preview-caption small")?.innerText ?? "";
const playhead = () => $<HTMLInputElement>('input[type="range"]');

/** Click "Générer et animer", wait for the new motion and let it play to the end. */
async function generateAndWatch(extraMs = 500, maxPlayMs = 6000) {
  const before = motionInfo();
  const btn = byText("button", "Générer et animer");
  await click(btn ?? null, 100);
  highlight($(".preview-card"));
  await waitFor(() => motionInfo() !== before && motionInfo().includes("images"), 90000).catch(() => undefined);
  await realSleep(200);
  const duration = Number(playhead()?.max ?? 3000);
  await realSleep(Math.min(maxPlayMs, Math.max(1500, duration)) + extraMs);
  highlight(null);
}

async function chooseEmotion(value: string) {
  const select = $<HTMLSelectElement>(".lsf-emotion select")!;
  await moveTo(select);
  highlight(select);
  setNative(select, value);
  await sleep(500);
  highlight(null);
}

// ------------------------------------------------------------------ scenes
type Scene = { id: string; run: () => Promise<void> };

const scenes: Scene[] = [
  {
    id: "intro",
    run: async () => {
      window.scrollTo({ top: 0 });
      await titleCard("Signa<em>Culture</em> AI", "Du français à la langue des signes française, signée par un avatar 3D", 2600);
      await say("", "Un avatar interprète, prêt à signer", "Respiration, regard et clignements : l’avatar reste vivant même au repos");
      await sleep(2400);
    },
  },
  {
    id: "question",
    run: async () => {
      await say("1 · TEXTE → LSF", "Une phrase française devient une phrase signée", "« Bonjour, comment ça va ? »");
      await typeInto(textArea(), "Bonjour, comment ça va ?");   // typed: example chips now sign at once with C0
      const panel = enginePanel();
      if (!panel.open) await click(panel.querySelector("summary"), 500);
      // C0 is the default engine; the word-order and lexicon scenes use B2, the generator scene switches to C0
      await click($('input[name="lsf-engine"][value="b2"]')?.closest("label") ?? null, 300);
      await say("1 · TEXTE → LSF", "Grammaire LSF : l’ordre des signes change", "Le mot interrogatif passe à la fin : BONJOUR · ALLER · COMMENT ?");
      await click(byText("button", /l’ordre LSF/) ?? null, 300);
      const order = await waitFor(() => $(".gloss-clause"), 15000).catch(() => null);
      highlight(order);
      await sleep(1700);
      highlight(null);
      await say("1 · TEXTE → LSF", "Visage grammatical : question ouverte", "Sourcils froncés et tête inclinée, comme une interprète");
      await generateAndWatch(700);
    },
  },
  {
    id: "negation",
    run: async () => {
      await say("2 · NÉGATION", "La négation se signe aussi avec la tête", "« Je ne veux pas boire de l’eau. »");
      await typeInto(textArea(), "Je ne veux pas boire de l'eau.");
      await generateAndWatch(500);
    },
  },
  {
    id: "emotion",
    run: async () => {
      await say("3 · ÉMOTION", "La même phrase, signée avec joie", "Gestes plus amples et plus vifs, sourire — la grammaire du visage reste prioritaire");
      await typeInto(textArea(), "Je visite Paris avec ma famille.");
      await chooseEmotion("happy");
      await generateAndWatch(600);
      await chooseEmotion("none");
    },
  },
  {
    id: "generator",
    run: async () => {
      await say("4 · GÉNÉRATEUR C0", "N’importe quelle phrase, même hors lexique", "Un modèle génératif entraîné sur des interprètes LSF produit le mouvement");
      await typeInto(textArea(), "Le musée est fermé demain.");
      await click($('input[name="lsf-engine"][value="c0"]')?.closest("label") ?? null, 300);
      await generateAndWatch(300, 5000);
      await click($('input[name="lsf-engine"][value="b2"]')?.closest("label") ?? null, 200);
    },
  },
  {
    id: "voice",
    run: async () => {
      await say("5 · VOIX", "Votre voix devient le texte", "Transcription Whisper et émotion de la voix (emotion2vec), entièrement en local");
      const input = $<HTMLInputElement>('input[type="file"][accept^="audio/"]')!;
      await moveTo(input.closest("label") ?? input.parentElement!);
      highlight(input.closest("label"));
      await attach(input, `${MEDIA}/voice.wav`, "voix.wav", "audio/wav");
      const use = await waitFor(() => byText("button", "Utiliser ce texte"), 60000).catch(() => null);
      highlight(null);
      if (use) {
        highlight($(".audio-transcript"));
        await sleep(2400);
        highlight(null);
        await click(use, 400);
        await generateAndWatch(300);
      }
    },
  },
  {
    id: "film",
    run: async () => {
      await say("6 · MODE FILM", "Un film et ses sous-titres, signés en synchronie", "Chaque réplique est traduite et signée à côté de l’image");
      await click(byText("button.nav-item", "Film") ?? null, 700);
      window.scrollTo({ top: 0 });
      const inputs = await waitFor(() => {
        const list = document.querySelectorAll<HTMLInputElement>(".film-inputs input[type=file]");
        return list.length >= 2 ? list : null;
      });
      await moveTo(inputs[0].closest("label")!);
      await attach(inputs[0], `${MEDIA}/film.mp4`, "conference.mp4", "video/mp4");
      await sleep(500);
      await moveTo(inputs[1].closest("label")!);
      await attach(inputs[1], `${MEDIA}/film.vtt`, "conference.fr.vtt", "text/vtt");
      await sleep(500);
      const filmB2 = [...document.querySelectorAll<HTMLInputElement>('input[name="film-engine"]')].find((r) => r.closest("label")?.innerText.startsWith("B2"));
      if (filmB2 && !filmB2.checked) await click(filmB2.closest("label"), 300);
      await click(byText("button", "Générer la piste LSF") ?? null, 300);
      await waitFor(() => {
        const s = $(".film-progress small")?.innerText ?? "";
        const m = s.match(/(\d+) \/ (\d+) sous-titres signés · (\d+) \/ (\d+) prêts/);
        return m && m[1] === m[2] && m[3] === m[4] && Number(m[2]) > 0;
      }, 120000).catch(() => undefined);
      const stage = $(".film-stage")!;
      stage.scrollIntoView({ behavior: FAST ? "auto" : "smooth", block: "center" });
      await sleep(900);
      await say("6 · MODE FILM", "L’avatar remplace l’interprète à l’écran", "Sous-titres français → LSF, réplique par réplique");
      cursor.style.opacity = "0";
      const video = $<HTMLVideoElement>(".film-video video")!;
      video.currentTime = 0;
      await video.play().catch(() => undefined);
      await realSleep(FAST ? 500 : 3000);
      hush();                                           // nothing in front of the film and the avatar while they play
      await realSleep(FAST ? 1000 : 8000);
      video.pause();
      cursor.style.opacity = "1";
    },
  },
  {
    id: "outro",
    run: async () => {
      hush();
      highlight(null);
      await titleCard("Signa<em>Culture</em> AI", "Texte, voix et films en LSF · tout fonctionne en local", 3500);
    },
  },
];

// ------------------------------------------------------------------ warm-up and start
/** Voice analysis reloads Whisper + emotion2vec per call (~45 s), and crashes the API once the C0 model is loaded
 * in the same process. The tour therefore replays the real analysis of the demo clip, computed beforehand with the same
 * endpoint and saved as demo-media/voice.analysis.json, when the UI sends that clip. */
let voiceAnalysis: Promise<string | null> | null = null;
function analyseVoiceOnce() {
  voiceAnalysis ??= fetch(`${MEDIA}/voice.analysis.json`).then((r) => (r.ok ? r.text() : null)).catch(() => null);
  return voiceAnalysis;
}
function replayVoiceAnalysis() {
  const real = window.fetch.bind(window);
  window.fetch = async (input: RequestInfo | URL, init?: RequestInit) => {
    const url = typeof input === "string" ? input : input instanceof URL ? input.href : input.url;
    const file = init?.body instanceof FormData ? init.body.get("file") : null;
    if (url.includes("/analyze-audio") && file instanceof File && file.name === "voix.wav") {
      const cached = await analyseVoiceOnce();
      if (cached) return new Response(cached, { status: 200, headers: { "Content-Type": "application/json" } });
    }
    return real(input, init);
  };
}

async function warmUp(status: (s: string) => void) {
  const post = (body: object) =>
    fetch("/api/lsf/generate", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body) }).catch(() => undefined);
  status("Chargement du lexique LSF…");
  await post({ text: "Bonjour, comment ça va ?", engine: "b2", order: "lsf", nonmanual: true });
  status("Chargement du générateur C0…");
  await post({ text: "Le musée est fermé demain.", engine: "c0", order: "french", nonmanual: true });
  await analyseVoiceOnce();
  status("Préparation des médias…");
  await Promise.all(["film.mp4", "film.vtt", "voice.wav"].map((f) => fetch(`${MEDIA}/${f}`).catch(() => undefined)));
  await waitFor(() => document.querySelector("canvas"), 60000).catch(() => undefined);
  await realSleep(1500);
}

export async function startDemoTour() {
  await waitFor(() => document.querySelector("textarea"), 60000);
  overlay();
  replayVoiceAnalysis();
  const gate = Object.assign(document.createElement("div"), { className: "demo-gate" });
  gate.innerHTML = `<div><b>Préparation de la démo…</b><small>Chargement des modèles</small></div>`;
  document.body.append(gate);
  await warmUp((s) => (gate.querySelector("small")!.textContent = s));
  gate.innerHTML = `<div><b>Démo prête</b><small>Lancez l’enregistrement d’écran, puis appuyez sur Entrée</small></div>`;
  await new Promise<void>((resolve) => {
    const onKey = (e: KeyboardEvent) => {
      if (e.key === "Enter") {
        window.removeEventListener("keydown", onKey);
        resolve();
      }
    };
    window.addEventListener("keydown", onKey);
    if (FAST) resolve();
  });
  gate.remove();
  if (document.activeElement instanceof HTMLElement) document.activeElement.blur();
  await realSleep(FAST ? 50 : 1200);
  const start = performance.now();
  const timings: Record<string, number> = {};
  const first = FROM ? Math.max(0, scenes.findIndex((s) => s.id === FROM)) : 0;
  for (const scene of scenes.slice(first)) {
    const t = performance.now();
    try {
      await scene.run();
    } catch (e) {
      console.warn(`demo scene ${scene.id} skipped:`, e);
    }
    timings[scene.id] = Math.round((performance.now() - t) / 100) / 10;
  }
  hush();
  cursor.style.opacity = "0";
  const total = Math.round((performance.now() - start) / 100) / 10;
  (window as unknown as { demoTimings: object }).demoTimings = { total, ...timings };
  console.info("demo timings (s)", { total, ...timings });
}
