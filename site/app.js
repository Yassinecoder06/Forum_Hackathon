// Showcase page: renders the real examples recorded by e2e/build-showcase.mjs and links into the live Studio.
const studioLink = (text) => `/studio/?text=${encodeURIComponent(text)}`;
// fr-FR groups with U+202F, which the display font lacks; use a regular no-break space.
const fmt = { format: (n) => new Intl.NumberFormat("fr-FR").format(n).replace(/\s/g, " ") };
// Studio exports are 1280x960 with the avatar at y 68-830 and a credits bar below;
// .frame crops to the avatar area (credits are shown on the page instead).
const framed = (video) => el("div", { className: "frame" }, video);
const inView = new IntersectionObserver((entries) => {
  for (const { target, isIntersecting } of entries) {
    if (isIntersecting && !window.matchMedia("(prefers-reduced-motion: reduce)").matches) target.play().catch(() => { target.controls = true; }); else target.pause();
  }
}, { threshold: 0.4 });
const el = (tag, props = {}, children = []) => {
  const node = Object.assign(document.createElement(tag), props);
  for (const child of [].concat(children)) node.append(child);
  return node;
};

function previewMedia(video, ex) {
  video.controls = true;
  video.muted = true;
  video.loop = true;
  video.playsInline = true;
  video.poster = ex.poster || ex.video.replace(/\.webm$/, ".jpg");
  const choices = ex.video_mp4 ? [[ex.video_mp4,"video/mp4"],[ex.video,"video/webm"]] : [[ex.video,"video/webm"]];
  video.removeAttribute("src");
  video.replaceChildren(...choices.map(([src,type]) => el("source",{src,type})));
  video.addEventListener("error", () => {
    const parent = video.closest(".example-media") || video.parentElement;
    if (!parent?.querySelector(".preview-error")) parent?.append(el("p",{className:"preview-error",textContent:"Aperçu indisponible sur ce navigateur. Ouvrez le Studio pour jouer les signes."}));
  });
  video.load();
  return video;
}

const DIAGRAMS = [
  ["01-overview", "Vue d'ensemble", "Une phrase en français devient une suite de signes réels, joués par l'avatar."],
  ["02-capture", "Capture", "Chaque signe est extrait une fois de sa vidéo source par MediaPipe ; une main absente n'est jamais inventée."],
  ["03-retargeting", "Avatar", "Les points capturés pilotent le squelette 3D en cinq étapes, sans modifier les données d'origine."],
  ["04-timeline", "Enchaînement", "Chaque clip est réduit à son geste utile, puis relié au suivant par une transition de 333 ms qui garde l'élan des bras ; la voix peut rythmer l'ensemble."],
  ["05-source-pose-avatar", "Trois vues", "Le même signe (CHAT) vu dans la vidéo source, dans les points capturés et sur l'avatar."],
  ["07-language-results", "Modèles de langue", "Les modèles testés ne traduisent pas encore : le dictionnaire reste la méthode fiable."],
  ["08-motion-quality", "Qualité", "Gains mesurés sur les transitions, les doigts, les mains et les sourcils."],
];

function showSource(sign) {
  const dialog = document.getElementById("source-dialog");
  document.getElementById("source-title").textContent = `${sign.gloss} · vidéo source`;
  const video = document.getElementById("source-video");
  video.src = sign.source_video;
  const credit = document.getElementById("source-credit");
  credit.replaceChildren(`${sign.credit || ""} · ${sign.license || ""} · `,
    el("a", { href: sign.source_page, target: "_blank", rel: "noopener", textContent: "page d'origine" }));
  dialog.showModal();
}
document.getElementById("source-dialog").addEventListener("close", () => {
  const video = document.getElementById("source-video");
  video.pause();
  video.removeAttribute("src");
  video.load();
});

function renderExample(ex) {
  const signs = ex.signs.map((s) =>
    el("li", {}, [
      el("button", { className: "chip", type: "button", textContent: s.gloss, title: `Voir la vidéo source de ${s.gloss}`, onclick: () => showSource(s) }),
      el("span", { className: "dur", textContent: `${(s.duration_ms / 1000).toFixed(1).replace(".", ",")} s` }),
    ]));
  const missing = ex.unknowns.length ? `Non signés : ${ex.unknowns.join(", ")}` : "Concepts du dictionnaire représentés";
  const video = previewMedia(el("video", {preload:"metadata",title:"Aperçu des mouvements capturés"}), ex);
  inView.observe(video);
  return el("article", { className: "example" }, [
    el("div", { className: "example-media" }, framed(video)),
    el("div", { className: "example-body" }, [
      el("h3", { textContent: `« ${ex.text} »` }),
      el("p", { className: "meta", textContent: `${ex.signs.length} signes · ${(ex.duration_ms / 1000).toFixed(1).replace(".", ",")} s · ${missing}` }),
      el("ul", { className: "signs" }, signs),
      el("div", { className: "example-actions" }, [
        el("a", { className: "button small", href: studioLink(ex.text), textContent: "Ouvrir dans le Studio" }),
      ]),
    ]),
  ]);
}

function renderTabs() {
  const tabs = document.getElementById("tabs");
  const img = document.getElementById("diagram-img");
  const caption = document.getElementById("diagram-caption");
  const select = (i) => {
    const [file, title, text] = DIAGRAMS[i];
    img.src = `/diagrams/${file}.png`;
    img.alt = `Diagramme : ${title}`;
    caption.textContent = text;
    tabs.querySelectorAll("button").forEach((b, j) => b.setAttribute("aria-selected", String(i === j)));
  };
  DIAGRAMS.forEach(([, title], i) => tabs.append(el("button", { type: "button", role: "tab", textContent: title, onclick: () => select(i) })));
  select(0);
}

async function main() {
  renderTabs();
  const data = await fetch("data/examples.json").then((r) => r.json());
  const featured = data.examples.find((e) => e.slug === "je-visite-paris-avec-ma-famille") || data.examples[0];
  previewMedia(document.getElementById("hero-video"), featured);
  document.getElementById("hero-text").textContent = `« ${featured.text} »`;
  document.getElementById("hero-glosses").replaceChildren(...featured.signs.map((s) => el("span", { className: "chip", textContent: s.gloss })));
  document.getElementById("gallery").replaceChildren(...data.examples.map(renderExample));
  document.getElementById("suggest").replaceChildren(...data.examples.map((e) => el("a", { href: studioLink(e.text), textContent: e.text })));
  try {
    const caps = await fetch("/api/capabilities").then((r) => r.json());
    if (caps.library_count) document.getElementById("stat-signs").textContent = fmt.format(caps.library_count);
    if (caps.dictasign?.total) document.getElementById("stat-dicta").textContent = fmt.format(caps.dictasign.total);
  } catch { /* static numbers remain */ }
}
main();
