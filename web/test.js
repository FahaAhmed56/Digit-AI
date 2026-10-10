// Run with: node web/test.js
// Checks the browser network gives the same probabilities as the Python one.
const fs = require("fs"), path = require("path"), vm = require("vm");
const ctx = {};
vm.createContext(ctx);
vm.runInContext(fs.readFileSync(path.join(__dirname, "model.js"), "utf8") + "\nthis.MODEL = MODEL;", ctx);
vm.runInContext(fs.readFileSync(path.join(__dirname, "nn.js"), "utf8") + "\nthis.predict = predict;", ctx);

const samples = JSON.parse(fs.readFileSync(path.join(__dirname, "test_samples.json"), "utf8"));
let maxDiff = 0, correct = 0;
for (const s of samples) {
  const p = ctx.predict(s.x);
  p.forEach((v, i) => { maxDiff = Math.max(maxDiff, Math.abs(v - s.probs[i])); });
  if (p.indexOf(Math.max(...p)) === s.y) correct++;
}
if (maxDiff > 1e-3) { console.error(`FAIL: JS and Python differ by ${maxDiff}`); process.exit(1); }
console.log(`✓ JS matches Python on ${samples.length} samples (max difference ${maxDiff.toExponential(1)}), ${correct}/${samples.length} correct`);
