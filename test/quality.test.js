/**
 * 🔴 THE QUALITY GATE — the test that would have caught the garbage.
 *
 * George trained `standard`, pressed "Sample from it", and got word-shaped noise.
 * Nothing threw, nothing was broken, and no existing test noticed: the model
 * trained, the loss fell, the samples were produced. It was garbage because the
 * preset stopped training while the loss was still about 2.5, and at that point a
 * character model has learned which characters are COMMON and not yet which
 * characters FOLLOW which — so it emits the right alphabet in the wrong order.
 *
 * The existing suite could not see this because it asserts MECHANISM (gradients
 * are right, the vocab round-trips, the shapes compose). This file asserts
 * OUTCOME: after its full configured number of steps, every preset must produce
 * text in which recognisable words appear.
 *
 * Slow by design — it trains each preset for its whole configured run, about
 * eight minutes for all three. `npm test` runs the fast structural checks in
 * `training.test.js`; run this with:
 *
 *     npm run test:quality              # every preset
 *     PRESET=quick npm run test:quality # just one, for a quick check
 *
 * It must pass before a deploy.
 *
 * ⚠️ CALIBRATION BOUND, stated so it is never claimed otherwise. This is measured
 * on a corpus of 8,389 characters. The presets train a FIXED number of steps, so a
 * much larger paste sees each character proportionally less often and its loss will
 * be higher than anything measured here. About 12,000 characters — the size that
 * produced the original report — is still comfortably inside the guarantee; a
 * 100,000-character book is not. **Scaling the step count to the corpus is the
 * obvious next improvement and has deliberately NOT been done**, because it would
 * make a large paste take proportionally longer and that is a product decision, not
 * a bug fix.
 */
import { test } from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';

import { Trainer, buildVocab, encode, preset, PRESETS, RECOGNISABLE_LOSS, plannedSteps } from '../site/llm.js';

/** A real prose document of about the size a visitor pastes (8,389 characters). */
const CORPUS = readFileSync(new URL('../README.md', import.meta.url), 'utf8');

/**
 * The share of a sample's 5+ character "words" that really occur in the corpus.
 *
 * This is the measurement that turns "that looks like garbage" into a number.
 * Measured against the loss on the same corpus:
 *
 *     loss 2.47 -> 3%     loss 1.65 -> 13%     loss 0.91 -> 31%
 *     loss 2.07 -> 4%     loss 1.36 -> 23%
 *     loss 1.94 -> 4%
 *
 * Anything at or below about 4% is noise. The floor is set at 12% — roughly
 * three times the noise level, and comfortably below what every preset achieves
 * once it is trained enough.
 */
function wordRatio(sample, corpus) {
  const words = sample.toLowerCase().match(/[a-z]{5,}/g) ?? [];
  if (!words.length) return 0;
  const hay = corpus.toLowerCase();
  return words.filter((w) => hay.includes(w)).length / words.length;
}

const REAL_WORD_FLOOR = 0.12;

/**
 * 🔴 A FIXED SEED, because the trainer starts from a random initialisation and a
 * gate that reports a different number every run is not a gate. The seed is
 * arbitrary — it is NOT chosen to flatter any preset: it was fixed after the
 * presets were chosen to clear the threshold with margin, and the first preset it
 * was tried on (`thorough`, at 800 steps) FAILED on it at loss 1.679 where an
 * earlier unseeded run had shown 1.310. That failure is the reason `thorough` is
 * now a 2-layer model trained twice as long. **A loss is a sample, not a value —
 * never set a threshold from one run.**
 */
const SEED = 20260918;

function trainAndSample(key) {
  const config = preset(key);
  const vocab = buildVocab(CORPUS);
  const data = encode(CORPUS, vocab);
  const trainer = new Trainer({ data, vocab, config, seed: SEED });

  // The steps the PAGE would really take — via the same `plannedSteps` the host
  // uses. Training more would be dishonest: a run capped in the browser can never
  // reach a longer one, so the cap has to be inside the thing being measured.
  const steps = plannedSteps(config, data.length);
  for (let i = 0; i < steps; i += 1) trainer.step();

  // Primed from a real line of the document, which is what the page's own
  // "Start it off" box is for, and what the measurements above were taken at.
  const seed = CORPUS.split('\n').find((l) => l.trim().length > 30)?.trim().slice(0, 24) ?? 'the';
  const samples = [0, 1, 2].map((i) =>
    trainer.sample({ prompt: seed, temperature: 0.7, topK: 40, length: 170, seed: 99 + i * 7919 })
  );
  return { trainer, samples, seed, steps };
}

const requested = process.env.PRESET;
const keys = requested ? [requested] : ['quick', 'standard', 'thorough'];

for (const key of keys) {
  test(`${key} trains until the output is recognisable, not just until the clock says stop`, () => {
    assert.ok(PRESETS[key], `unknown preset ${key}`);
    const { trainer, samples, seed, steps } = trainAndSample(key);

    // 🔴 The cap is checked FIRST, because it used to make every other number here
    // meaningless: at 40 epochs each preset was cut to 605 steps on the built-in
    // corpus, so what the preset asked for did not decide anything.
    assert.equal(
      steps,
      PRESETS[key].steps,
      `${key} ran ${steps} steps, not the ${PRESETS[key].steps} it asks for — the epoch cap ` +
        `is throttling the preset, which is exactly how the presets came to be overridden`
    );

    const ratio = samples.reduce((sum, s) => sum + wordRatio(s, CORPUS), 0) / samples.length;
    const detail =
      `${key}: ${steps} steps, loss ${trainer.smoothedLoss.toFixed(3)} (must be < ${RECOGNISABLE_LOSS}), ` +
      `real words ${(ratio * 100).toFixed(0)}% (must be >= ${REAL_WORD_FLOOR * 100}%), ` +
      `primed with ${JSON.stringify(seed)}\n` +
      samples.map((s) => '    * ' + s.replace(/\n/g, ' / ').slice(0, 160)).join('\n');
    console.log('\n' + detail + '\n');

    assert.ok(
      trainer.smoothedLoss < RECOGNISABLE_LOSS,
      `the loss is still ${trainer.smoothedLoss.toFixed(3)} — above the point where words ` +
        `appear. The preset is stopping too early, which is exactly the bug this guards.\n${detail}`
    );
    assert.ok(
      ratio >= REAL_WORD_FLOOR,
      `only ${(ratio * 100).toFixed(0)}% of the sampled words occur in the corpus — that is ` +
        `noise, whatever the loss says.\n${detail}`
    );
  });
}
