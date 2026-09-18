/**
 * What does this model actually do with PROSE?
 *
 *   node tools/probe-prose.mjs
 *
 * The page's own advice is "a list of names, a list of products, a script with
 * repeated turns" — small and highly structured. This runs the opposite case on
 * purpose: eight short diary entries, 269 words, and prints what comes back.
 * That is the honest answer to "can I paste my diary and ask it questions?", and
 * it is measured here rather than asserted in prose.
 *
 * Measured 18 Sep 2026, `standard` preset (2 layers, 75,828 parameters): loss
 * 1.414 against a random start of log(52) = 3.95, and the output is the SHAPE of
 * a diary — day headers, entry breaks, "I", "the", "day", "March" — around words
 * that are mostly not words. The model learned the form and not the language,
 * which is exactly what its size allows.
 *
 * ⚠️ THAT MEASUREMENT WAS TAKEN WHEN `standard` TRAINED 250 STEPS. It now trains
 * its full 1,500, so a run of this probe takes about two minutes instead of
 * twenty-seven seconds and the text is markedly better. The honest limit it
 * demonstrates has not moved — a model this size on prose of this kind still
 * learns form rather than meaning — but the numbers below are historical, and
 * `tools/probe-curve.mjs` is the rig to re-measure them.
 *
 * Not part of the test suite: it prints a judgement, not a pass or a fail.
 */
import { Trainer, preset, buildVocab, encode, lineStartToken } from '../site/llm.js';

const DIARY = `Monday 3 March
Slept badly again. Up at six, coffee, then the drive in. The meeting ran long and I said almost nothing, which is becoming a habit. Home by seven, made pasta, watched nothing, bed early.

Tuesday 4 March
Rain all day. I took the long way home past the river. There is a heron that stands in the same spot every evening, and I have started to look for it. Small thing. It helps.

Wednesday 5 March
Called my mother. She talked about the garden for twenty minutes and I let her. I should call more often. Work was quiet, which I needed.

Thursday 6 March
Bad day. Nothing specific, just the weight of it. I did not sleep until two. Told myself I would go for a run in the morning and I know I will not.

Friday 7 March
Better. Finished the report, sent it at four, and left before anyone else. The heron was there. Stopped at the market and bought too much fruit.

Saturday 8 March
Long walk in the morning, then nothing at all for the rest of the day. Read half a book. This is the first Saturday in weeks that did not feel borrowed.

Sunday 9 March
Rained again. Cleaned the kitchen, fixed the shelf that has been loose since January, and felt absurdly pleased about it. Called my mother, and this time I listened.

Monday 10 March
Back to it. Slept badly again. The drive in was slow and the meeting ran long. I am noticing a pattern in how I write these, and the pattern is the point.`;

const config = preset('standard');
const vocab = buildVocab(DIARY);
const data = encode(DIARY, vocab);
console.log(`\ncorpus: ${DIARY.length} characters (${DIARY.split(/\s+/).length} words), vocabulary of ${vocab.size} characters`);
console.log(`preset: ${config.nLayer} layer(s), dModel ${config.dModel}, context ${config.blockSize}, ${config.steps} steps\n`);

const trainer = new Trainer({ data, vocab, config });
console.log('parameters:', trainer.paramCount);

const started = Date.now();
for (let i = 0; i < config.steps; i += 1) {
  const { loss, step } = trainer.step();
  if (step % 50 === 0 || step === 1) {
    console.log(`  step ${String(step).padStart(3)}  loss ${loss.toFixed(3)}`);
  }
}
console.log(`\ntrained in ${((Date.now() - started) / 1000).toFixed(1)}s, smoothed loss ${trainer.smoothedLoss.toFixed(3)}`);
console.log(`(a model guessing among ${vocab.size} characters starts near log(${vocab.size}) = ${Math.log(vocab.size).toFixed(2)})\n`);

for (const [label, options] of [
  ['from "Monday"', { prompt: 'Monday', temperature: 0.7, topK: 40, length: 140 }],
  ['from "Bad day"', { prompt: 'Bad day', temperature: 0.7, topK: 40, length: 140 }],
  ['from "Slept"', { prompt: 'Slept', temperature: 0.7, topK: 40, length: 140 }],
  ['from nothing at all', { prompt: '', temperature: 0.9, topK: 40, length: 140 }],
]) {
  console.log(`--- ${label} ---`);
  for (let i = 0; i < 3; i += 1) {
    console.log('  * ' + trainer.sample({ ...options, seed: 1000 + i * 7919 }).replace(/\n/g, ' / '));
  }
  console.log();
}
console.log('the first token the model is handed:', lineStartToken(data, vocab));
