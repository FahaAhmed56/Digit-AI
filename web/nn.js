// The same forward pass as training/network.py, in plain JavaScript.
function predict(input) {                       // input: 64 numbers in 0..1
  const { W1, b1, W2, b2 } = MODEL;
  const hidden = b1.map((b, j) => Math.max(0, input.reduce((s, x, i) => s + x * W1[i][j], b)));   // ReLU
  const logits = b2.map((b, k) => hidden.reduce((s, h, j) => s + h * W2[j][k], b));
  const max = Math.max(...logits);
  const exps = logits.map(z => Math.exp(z - max));
  const sum = exps.reduce((a, b) => a + b, 0);
  return exps.map(e => e / sum);                // softmax probabilities for digits 0..9
}

const BLUR = 1.5;   // must match BLUR in training/preprocess.py
var DILATE = 2;     // pixels of stroke thickening (app only)

// Turn the drawing into 64 numbers using the SAME steps as training/preprocess.py:
// crop to the ink, fit the longest side to 28px inside a 32x32 box,
// blur slightly, average 4x4 blocks into an 8x8 grid, then scale so the darkest cell is 1.
function toInput(canvas) {
  const W = canvas.width, H = canvas.height;
  const px = canvas.getContext("2d").getImageData(0, 0, W, H).data;
  let minX = W, minY = H, maxX = -1, maxY = -1;
  for (let y = 0; y < H; y++) for (let x = 0; x < W; x++) {
    if (px[(y * W + x) * 4 + 3] > 25) {          // alpha > 0.1, same threshold as Python
      if (x < minX) minX = x; if (x > maxX) maxX = x;
      if (y < minY) minY = y; if (y > maxY) maxY = y;
    }
  }
  if (maxX < 0) return null;                    // nothing drawn

  const bw = maxX - minX + 1, bh = maxY - minY + 1;
  const scale = 28 / Math.max(bw, bh);
  const dw = Math.max(1, Math.round(bw * scale)), dh = Math.max(1, Math.round(bh * scale));
  const small = document.createElement("canvas");
  small.width = small.height = 32;
  const sctx = small.getContext("2d");
  sctx.imageSmoothingQuality = "high";
  sctx.drawImage(canvas, minX, minY, bw, bh, Math.floor((32 - dw) / 2), Math.floor((32 - dh) / 2), dw, dh);

  // App-only step: the dataset's strokes are thick (about 2 of the 8 cells), but a pen line shrunk
  // into this box is only about 3px wide. Dilation (each pixel takes the max of its neighbours)
  // thickens it to match. Training images skip this step because they are already thick.
  if (DILATE > 0) {
    const img = sctx.getImageData(0, 0, 32, 32), a = img.data, src = a.slice();
    for (let y = 0; y < 32; y++) for (let x = 0; x < 32; x++) {
      let m = 0;
      for (let dy = -DILATE; dy <= DILATE; dy++) for (let dx = -DILATE; dx <= DILATE; dx++) {
        const yy = y + dy, xx = x + dx;
        if (yy >= 0 && yy < 32 && xx >= 0 && xx < 32) m = Math.max(m, src[(yy * 32 + xx) * 4 + 3]);
      }
      a[(y * 32 + x) * 4 + 3] = m;
    }
    sctx.putImageData(img, 0, 0);
  }

  // Same Gaussian blur as training/preprocess.py
  const blurred = document.createElement("canvas");
  blurred.width = blurred.height = 32;
  const bctx = blurred.getContext("2d");
  bctx.filter = `blur(${BLUR}px)`;
  bctx.drawImage(small, 0, 0);

  const s = bctx.getImageData(0, 0, 32, 32).data;
  const grid = [];
  for (let gy = 0; gy < 8; gy++) for (let gx = 0; gx < 8; gx++) {
    let total = 0;
    for (let y = 0; y < 4; y++) for (let x = 0; x < 4; x++)
      total += s[((gy * 4 + y) * 32 + gx * 4 + x) * 4 + 3] / 255;
    grid.push(total / 16);
  }
  const max = Math.max(...grid);
  return grid.map(v => v / max);
}

if (typeof module !== "undefined") module.exports = { predict };
