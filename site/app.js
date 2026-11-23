// Landing page: example sentences that open in the live Studio, and the pipeline diagrams.
const studioLink = (text) => `/studio/?text=${encodeURIComponent(text)}`;
const el = (tag, props = {}, children = []) => {
  const node = Object.assign(document.createElement(tag), props);
  for (const child of [].concat(children)) node.append(child);
  return node;
};

const SENTENCES = [
  "Je peux vous le redire une troisième fois, si vous le souhaitez.",
  "C'est évidemment un signe extrêmement encourageant.",
  "C'est une réussite concrète dont bénéficient chaque jour des millions d'enfants.",
  "Néanmoins, nous ne baissons pas la garde.",
];

const DIAGRAMS = [
  ["01-overview", "Vue d'ensemble", "Une phrase en français devient une suite de signes, jouée par l'avatar."],
  ["02-capture", "Capture", "Chaque signe du lexique est extrait de sa vidéo source par MediaPipe, mains et visage compris."],
  ["03-retargeting", "Avatar", "Les points capturés ou générés pilotent le squelette 3D de l'avatar."],
  ["04-timeline", "Enchaînement", "Chaque signe est relié au suivant par une transition qui garde l'élan des bras ; la voix peut rythmer l'ensemble."],
  ["05-source-pose-avatar", "Trois vues", "Le même signe vu dans la vidéo source, dans les points capturés et sur l'avatar."],
  ["08-motion-quality", "Qualité", "Gains mesurés sur les transitions, les doigts, les mains et les sourcils."],
];

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

document.getElementById("suggest").replaceChildren(...SENTENCES.map((t) => el("a", { href: studioLink(t), textContent: t })));
renderTabs();
const hero = document.getElementById("hero-video");
if (window.matchMedia("(prefers-reduced-motion: reduce)").matches) { hero.pause(); hero.controls = true; }
