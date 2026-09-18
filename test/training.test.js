import { test } from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';

import {
  Trainer,
  buildVocab,
  encode,
  parameterCount,
  preset,
  PRESETS,
  RECOGNISABLE_LOSS,
  MAX_EPOCHS,
  plannedSteps,
} from '../site/llm.js';

/* ------------------------------------------------------------------ *
 * The failure these tests exist to prevent
 * ------------------------------------------------------------------ *
 *
 * George trained the `standard` preset, pressed "Sample from it", and got
 * word-shaped noise. The transformer was not broken. THE STEP COUNT was: the
 * presets had been chosen to fit a stopwatch — 600 / 250 / 300 steps — which
 * left every one of them at a smoothed loss between 1.9 and 2.5.
 *
 * Measured 18 Sep 2026 on `README.md` as the corpus, sampling primed from a real
 * line at temperature 0.7 with top-k 40. The loss predicts the output sharply:
 *
 *     loss 2.47  ->   3% of the words in the sample occur in the corpus
 *     loss 2.07  ->   4%
 *     loss 1.94  ->   4%
 *     loss 1.65  ->  13%
 *     loss 1.36  ->  23%
 *     loss 0.91  ->  31%
 *
 * So "the text is garbage" is not a matter of taste — it is a loss above about
 * 1.6 — and it can be asserted. This file holds the fast half of that assertion
 * (the configuration is sufficient); `quality.test.js` holds the slow half (the
 * presets actually reach the loss when trained).
 */

/**
 * The smallest step count each SHAPE reaches `RECOGNISABLE_LOSS` at.
 *
 * All four numbers are measured, not chosen. On `README.md` (8,389 characters):
 *
 *   quick    2,000 steps -> loss 1.647 (just above the line)
 *            3,000 steps -> loss 1.567 on the fixed seed — a 0.033 margin, too thin
 *            4,000 steps -> loss 1.362 (comfortably below)
 *   standard 1,000 steps -> loss 1.567 (marginal)
 *            1,500 steps -> loss 1.179
 *   thorough   600 steps -> loss 1.718 (still noise)
 *              800 steps -> loss 1.310 on one seed and 1.679 on another — it does
 *                          not converge in 800 steps, which is why the 3-layer
 *                          shape was dropped for a 2-layer one trained 3,000 steps
 *
 * `quick` uses a bigger margin than its measurement strictly needs because it
 * has the smallest model and therefore the least headroom on unfamiliar text.
 */
const MIN_STEPS = { quick: 4000, standard: 1500, thorough: 3000 };

test('every preset is trained long enough to form words, not just characters', () => {
  for (const [key, minimum] of Object.entries(MIN_STEPS)) {
    assert.ok(
      PRESETS[key].steps >= minimum,
      `${key} trains ${PRESETS[key].steps} steps, below the measured minimum of ${minimum} — ` +
        `at that point the loss is still above ${RECOGNISABLE_LOSS} and the output is noise`
    );
  }
});

test('a preset that offers more costs more time and does more training', () => {
  const order = ['quick', 'standard', 'thorough'];
  const work = (c) => c.steps * parameterCount(c, 84);
  for (let i = 1; i < order.length; i += 1) {
    const cheaper = PRESETS[order[i - 1]];
    const dearer = PRESETS[order[i]];
    assert.ok(
      dearer.seconds > cheaper.seconds,
      `${order[i]} must take longer than ${order[i - 1]} (${dearer.seconds}s vs ${cheaper.seconds}s)`
    );
    // WORK, not size. A preset can offer more either by being a bigger model or by
    // training the same model longer, and `thorough` now does the second: it is
    // `standard`'s shape trained twice as long. The measurement is why — the
    // 3-layer shape costs 389 ms a step and did NOT reach the threshold in 800 of
    // them (loss 1.679), while the 2-layer shape reaches 0.601 in 3,000 steps and
    // 254 s. More training beat a bigger model, so the invariant has to be work.
    assert.ok(
      work(dearer) > work(cheaper),
      `${order[i]} must do more training work than ${order[i - 1]} ` +
        `(${work(dearer).toLocaleString()} vs ${work(cheaper).toLocaleString()})`
    );
  }
});

test('the advertised seconds are in the right order of magnitude, not optimistic', () => {
  // Measured throughput on a desktop CPU: about 87 steps/s for the 1-layer shape
  // and about 11.8 for the 2-layer one. A preset may not claim to finish faster
  // than the measured rate allows — that is how "about fifteen seconds" ended up
  // describing a model that had barely started learning.
  const measuredStepsPerSecond = { quick: 87, standard: 11.8, thorough: 11.8 };
  for (const [key, rate] of Object.entries(measuredStepsPerSecond)) {
    const honest = PRESETS[key].steps / rate;
    assert.ok(
      PRESETS[key].seconds >= honest * 0.8,
      `${key} claims ${PRESETS[key].seconds}s for ${PRESETS[key].steps} steps, but ` +
        `measured throughput needs about ${honest.toFixed(0)}s`
    );
  }
});

/* ------------------------------------------------------------------ *
 * Sampling defaults
 * ------------------------------------------------------------------ */

test('sample() truncates by default rather than drawing from the whole vocabulary', () => {
  // The page shipped `topK = 0`, which turns truncation OFF, so the long tail of
  // near-impossible characters was drawn constantly and the text was noisier for
  // no gain. Asserted by identity: with a fixed seed the model is deterministic,
  // so the no-argument call must equal the explicit good defaults.
  const ascii = Array.from({ length: 95 }, (_, i) => String.fromCharCode(32 + i)).join('');
  const text = ascii.repeat(40);
  const vocab = buildVocab(text);
  const data = encode(text, vocab);
  const config = {
    ...preset('quick'),
    nLayer: 1, nHead: 2, dModel: 8, dFF: 16,
    blockSize: 8, batchSize: 1, steps: 1, weightDecay: 0,
  };
  assert.ok(vocab.size > 40, 'this corpus must be wide enough for top-k 40 to bite');
  const trainer = new Trainer({ data, vocab, config, seed: 7 });

  const byDefault = trainer.sample({ seed: 5 });
  const explicit = trainer.sample({ temperature: 0.8, topK: 20, length: 240, seed: 5 });
  const untruncated = trainer.sample({ temperature: 0.8, topK: 0, length: 240, seed: 5 });

  assert.equal(byDefault, explicit, 'the default must be temperature 0.8, top-k 20, length 240');
  assert.notEqual(
    byDefault,
    untruncated,
    'and it must not be the old top-k 0 — on this wide vocabulary the two differ'
  );
});

test('the sample controls the page offers cannot silently disable truncation', () => {
  // The slider is the visitor's control; its default must agree with the model's.
  const html = readFileSync(
    new URL('../site/index.html', import.meta.url),
    'utf8'
  );
  const slider = html.match(/id="topK"[^>]*value="(\d+)"/) ?? html.match(/value="(\d+)"[^>]*id="topK"/);
  assert.ok(slider, 'the top-k slider must declare a default value');
  assert.ok(
    Number(slider[1]) > 0,
    `the top-k slider defaults to ${slider[1]}, which switches truncation off`
  );
});

/* ------------------------------------------------------------------ *
 * The run-length cap — the SECOND half of the same bug
 * ------------------------------------------------------------------ */

test('the epoch cap does not override the preset on a realistic corpus', () => {
  // 🔴 The presets were not the only thing deciding how long a run was. The host
  // capped every run at 40 passes over the corpus, computed from the CORPUS and
  // not from the preset:
  //
  //     steps = min(config.steps, (dataLength - 1) * 40 / (batchSize * blockSize))
  //
  // On the built-in 4,359-character name list that is **605 steps for every
  // preset**, so the `quick` preset's 600 steps were not even its own number —
  // and raising it to 3,000 would have changed nothing whatsoever. This is the
  // assertion that catches that, and it is why `plannedSteps` exists.
  assert.ok(MAX_EPOCHS >= 400, `the epoch ceiling is ${MAX_EPOCHS}, low enough to decide the run`);

  const realistic = [
    ['the built-in name list', 4359],
    ['a 12,000-character paste, the size that produced the garbage', 11939],
    ['a 30,000-character document', 30000],
  ];
  for (const [label, length] of realistic) {
    for (const key of Object.keys(MIN_STEPS)) {
      assert.equal(
        plannedSteps(PRESETS[key], length),
        PRESETS[key].steps,
        `${key} on ${label} is capped to fewer than the ${PRESETS[key].steps} steps it asks for — ` +
          `the run-length cap is deciding how well the model learns again`
      );
    }
  }

  // The ceiling must still exist, or a one-paragraph corpus would be read an
  // unbounded number of times and the progress estimate would be nonsense.
  assert.ok(
    plannedSteps(PRESETS.quick, 300) < PRESETS.quick.steps,
    'a very short corpus must still be capped'
  );
});
