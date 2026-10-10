const pad = document.getElementById("pad");
const ctx = pad.getContext("2d");
const preview = document.getElementById("preview").getContext("2d");
let drawing = false, last = null;

ctx.lineWidth = 22; ctx.lineCap = "round"; ctx.lineJoin = "round"; ctx.strokeStyle = "#111";

// Bars for each digit
const bars = document.getElementById("bars");
bars.innerHTML = [...Array(10).keys()].map(d =>
  `<div class="bar"><span>${d}</span><div class="track"><div class="fill" id="f${d}"></div></div><em id="p${d}">0%</em></div>`).join("");

function pos(e) {
  const r = pad.getBoundingClientRect(), t = e.touches ? e.touches[0] : e;
  return { x: (t.clientX - r.left) * pad.width / r.width, y: (t.clientY - r.top) * pad.height / r.height };
}
function start(e) { drawing = true; last = pos(e); draw(e); }
function draw(e) {
  if (!drawing) return;
  e.preventDefault();
  const p = pos(e);
  ctx.beginPath(); ctx.moveTo(last.x, last.y); ctx.lineTo(p.x, p.y); ctx.stroke();
  last = p;
  update();
}
function stop() { drawing = false; }

function update() {
  const input = toInput(pad);
  preview.clearRect(0, 0, 96, 96);
  if (!input) return reset();
  input.forEach((v, i) => {                       // draw the 8x8 grid preview
    const g = Math.round(255 - v * 255);
    preview.fillStyle = `rgb(${g},${g},${g})`;
    preview.fillRect((i % 8) * 12, Math.floor(i / 8) * 12, 12, 12);
  });
  const probs = predict(input);
  const best = probs.indexOf(Math.max(...probs));
  document.getElementById("guess").textContent = best;
  document.getElementById("conf").textContent = `${(probs[best] * 100).toFixed(1)}% confident`;
  probs.forEach((p, d) => {
    document.getElementById(`f${d}`).style.width = `${p * 100}%`;
    document.getElementById(`f${d}`).classList.toggle("top", d === best);
    document.getElementById(`p${d}`).textContent = `${Math.round(p * 100)}%`;
  });
}
function reset() {
  document.getElementById("guess").textContent = "?";
  document.getElementById("conf").textContent = "Start drawing";
  for (let d = 0; d < 10; d++) {
    document.getElementById(`f${d}`).style.width = "0";
    document.getElementById(`p${d}`).textContent = "0%";
  }
}

pad.addEventListener("mousedown", start); pad.addEventListener("mousemove", draw);
window.addEventListener("mouseup", stop);
pad.addEventListener("touchstart", start, { passive: false }); pad.addEventListener("touchmove", draw, { passive: false });
pad.addEventListener("touchend", stop);
document.getElementById("clear").addEventListener("click", () => { ctx.clearRect(0, 0, pad.width, pad.height); preview.clearRect(0, 0, 96, 96); reset(); });
reset();
