const get = selector => document.querySelector(selector);
const video = get("#video");
const search = get("#search");
const split = get("#split");
const samples = get("#samples");
let rows = [];
let selected;
let annotationButtons = [];
const normalize = value => value.toLowerCase().normalize("NFD").replace(/\p{Diacritic}/gu, "");
const eventName = event => event.gloss || event.lemma || "";
const seconds = ms => (ms / 1000).toFixed(2) + " s";

function seek(ms) {
  if (!selected) return;
  const position = Math.max(0, (ms - selected.clip_start_ms) / 1000);
  if (video.readyState) video.currentTime = Math.min(video.duration, position);
  else video.addEventListener("loadedmetadata", () => {
    video.currentTime = Math.min(video.duration, position);
  }, {once:true});
}

function updateTime() {
  if (!selected) return;
  const now = Math.round(video.currentTime * 1000) + selected.clip_start_ms;
  get("#time").textContent = `Source ${seconds(now)} · Translation ${seconds(selected.interval_ms[0])}–${seconds(selected.interval_ms[1])}`;
  for (const [button, event] of annotationButtons) {
    button.setAttribute("aria-pressed", String(event.start_ms <= now && now < event.end_ms));
  }
}

function show(row) {
  selected = row;
  video.src = row.clip;
  get("#caption").textContent = row.french_text;
  get("#tiers").replaceChildren();
  annotationButtons = [];
  for (const [name, events] of Object.entries(row.hand_tiers)) {
    const label = document.createElement("strong");
    label.textContent = name === "left_hand" ? "Left hand" : name === "right_hand" ? "Right hand" : name;
    const lane = document.createElement("div");
    lane.className = "lane";
    for (const event of events.filter(event => eventName(event))) {
      const button = document.createElement("button");
      button.type = "button";
      button.textContent = `${eventName(event)} · ${seconds(event.start_ms)}`;
      button.addEventListener("click", () => seek(event.start_ms));
      annotationButtons.push([button, event]);
      lane.append(button);
    }
    if (lane.childElementCount) get("#tiers").append(label, lane);
  }
  updateTime();
}

function refresh() {
  const query = normalize(search.value.trim());
  const filtered = rows.filter(row => (!split.value || row.split === split.value) &&
    normalize(row.french_text + " " + Object.values(row.hand_tiers).flat().map(eventName).join(" ")).includes(query));
  get("#count").textContent = `${filtered.length} of ${rows.length} excerpts`;
  get("#detail").hidden = !filtered.length;
  get("#empty").hidden = !!filtered.length;
  if (!filtered.length) video.pause();
  else if (!filtered.includes(selected)) show(filtered[0]);
  samples.replaceChildren(...filtered.map(row => {
    const button = document.createElement("button");
    const tag = document.createElement("small");
    tag.textContent = row.split;
    button.type = "button";
    button.textContent = row.french_text;
    button.append(tag);
    button.setAttribute("aria-pressed", String(row === selected));
    button.addEventListener("click", () => {
      show(row);
      for (const item of samples.children) item.setAttribute("aria-pressed", String(item === button));
    });
    return button;
  }));
}

video.addEventListener("timeupdate", updateTime);
video.addEventListener("seeked", updateTime);
get("#jump").addEventListener("click", () => seek(selected.interval_ms[0]));
search.addEventListener("input", refresh);
split.addEventListener("change", refresh);
fetch("data.json").then(response => {
  if (!response.ok) throw Error("Dataset unavailable");
  return response.json();
}).then(data => {
  rows = data;
  for (const name of new Set(rows.map(row => row.split))) split.add(new Option(name, name));
  refresh();
}).catch(() => { get("#count").textContent = "Local dataset sample is unavailable."; });
