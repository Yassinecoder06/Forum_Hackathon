const player = document.querySelector("#motion");
const buttons = [...document.querySelectorAll("[data-sample]")];
const methods = {
  model: "Trained motion decoder",
  phase_template: "Average training motion",
  retrieval_shrunk_top3: "Blended caption retrieval",
  retrieval_top1: "Nearest-caption retrieval"
};

function selectExample(data, sample) {
  const example = data.samples[sample];
  for (const button of buttons) button.setAttribute("aria-pressed", String(button.dataset.sample === sample));
  player.classList.toggle("demo", sample === "demo");
  player.classList.toggle("training", sample.startsWith("train_"));
  player.poster = example.poster;
  player.src = example.video;
  player.setAttribute("aria-label", example.label);
  document.querySelector("#caption-label").textContent = sample === "demo" ? "French model input" :
    sample.startsWith("train_") ? "Exact training caption" : "French subtitle";
  document.querySelector("#caption").textContent = example.caption;
  document.querySelector("#playback-note").textContent = example.note;
}

async function start() {
  const response = await fetch("assets/results.json");
  if (!response.ok) throw Error("Results unavailable");
  const data = await response.json();
  const group = document.querySelector(".options");
  Object.keys(data.samples).filter(sample => sample.startsWith("train_")).forEach((sample, i) => {
    const button = document.createElement("button");
    button.type = "button";
    button.dataset.sample = sample;
    button.textContent = `Training example ${i + 1}`;
    group.insertBefore(button, group.querySelector('[data-sample="demo"]'));
    buttons.push(button);
  });
  for (const button of buttons) button.addEventListener("click", () => selectExample(data, button.dataset.sample));
  selectExample(data, "new_test");
  const table = document.querySelector("#scores");
  for (const [id, label] of Object.entries(methods)) {
    const row = table.insertRow();
    row.classList.toggle("best", id === "phase_template");
    for (const value of [label, data.scores.test[id].toFixed(4), data.scores.new_test[id].toFixed(4)]) {
      row.insertCell().textContent = value;
    }
  }
}
start().catch(() => { document.querySelector("#caption").textContent = "Local experiment assets are unavailable."; });
