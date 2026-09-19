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
 * The smallest step count each SHAPE reaches coherence at.
 *
 * Measured on `README.md` (1,945 WORD tokens, 939-word vocabulary). The second
 * number is the share of a sample's adjacent word pairs that really occur in the
 * corpus — word salad scores near zero, memorised text near one:
 *
 *   quick    100 steps -> loss 3.189  pairs  86%
 *            250 steps -> loss 1.048  pairs  97%
 *            500 steps -> loss 0.235  pairs 100%
 *   standard 100 steps -> loss 2.620  pairs  91%
 *            250 steps -> loss 0.451  pairs  98%
 *            500 steps -> loss 0.127  pairs 100%
 *
 * And the CHARACTER level has its own measured minimums, because it is not the
 * same problem. On the built-in 653-name list — where every item occurs once — the
 * level is character and the corpus is 4,359 tokens:
 *
 *   quick    4,000 steps -> loss 0.810  29 of 36 lines were NEW names
 *   standard 1,500 steps -> loss 0.572  26 new, one repeated four times
 *   thorough 3,000 steps -> loss 0.364  16 new, and mostly reciting the real names
 */
const PRESET_KEYS = ['quick', 'standard', 'thorough'];

const MIN_STEPS = {
  word: { quick: 500, standard: 500, thorough: 1500 },
  char: { quick: 4000, standard: 1500, thorough: 3000 },
};

test('every preset is trained long enough for the level it is used at', () => {
  for (const [level, minimums] of Object.entries(MIN_STEPS)) {
    for (const [key, minimum] of Object.entries(minimums)) {
      const steps = PRESETS[key].levels[level].steps;
      assert.ok(
        steps >= minimum,
        `${key} at ${level} level trains ${steps} steps, below the measured minimum of ${minimum} — ` +
          `at that point the loss is still above ${RECOGNISABLE_LOSS[level]} and the output is noise`
      );
    }
  }
});

/*
 * 🔴 THE TWO LEVELS PULL IN OPPOSITE DIRECTIONS, AND ONLY ONE OF THEM IS "MORE IS
 * BETTER". This is the single most surprising measured thing in this project, so it
 * is asserted rather than left in a comment.
 *
 * At WORD level, more training is strictly better — that is the whole story of the
 * garbage bug, and the work invariant below holds.
 *
 * At CHARACTER level on a list, it is the opposite. Measured on the 653-name
 * corpus: quick (small model, 4,000 steps) invented 29 of 36 lines; thorough (bigger
 * model, 3,000 steps, loss 0.364) invented 16 and spent the rest reciting real names
 * with stutters like "Molll y" and "Moseses". A bigger model on a small list just
 * memorises it faster, so the loss keeps falling while the thing the visitor wants
 * disappears. **Do not "fix" the character presets by raising the step counts.**
 */
test('a preset that offers more does more training work — at WORD level', () => {
  const work = (c) => c.steps * parameterCount(c, 84);
  for (let i = 1; i < PRESET_KEYS.length; i += 1) {
    const cheaper = preset(PRESET_KEYS[i - 1], 'word');
    const dearer = preset(PRESET_KEYS[i], 'word');
    assert.ok(
      dearer.seconds > cheaper.seconds,
      `${PRESET_KEYS[i]} must take longer than ${PRESET_KEYS[i - 1]} ` +
        `(${dearer.seconds}s vs ${cheaper.seconds}s)`
    );
    // WORK, not size. A preset can offer more either by being a bigger model or by
    // training the same model longer, and `thorough` does the second: it is
    // `standard`'s shape trained three times as long. More training beat a bigger
    // model on measurement, so the invariant has to be work.
    assert.ok(
      work(dearer) > work(cheaper),
      `${PRESET_KEYS[i]} must do more training work than ${PRESET_KEYS[i - 1]} ` +
        `(${work(dearer).toLocaleString()} vs ${work(cheaper).toLocaleString()})`
    );
  }
});

test('the character presets that get WORSE on a list say so on the chip', () => {
  // The finding above, locked in so a later tidy-up cannot quietly delete the
  // warning and leave a five-minute preset that recites where a one-minute preset
  // invents. A visitor who is not told will read the worse output as a broken demo.
  assert.ok(
    PRESETS.thorough.levels.char.note?.includes('recites'),
    'thorough at character level is measured WORSE than quick on a list, and its chip must say so'
  );
  assert.ok(
    PRESETS.quick.levels.char.note,
    'quick at character level is measured the one that invents most, and its chip should say so'
  );
});

test('the advertised seconds are in the right order of magnitude, not optimistic', () => {
  // Measured throughput on a desktop CPU, per level — the 1-layer shape is far
  // faster than the 2-layer one, and character tokens are cheaper than word tokens
  // (a 53-token vocabulary against a 939-token one).
  const measuredStepsPerSecond = {
    word: { quick: 29, standard: 7, thorough: 7 },
    char: { quick: 77, standard: 11, thorough: 10 },
  };
  for (const [level, rates] of Object.entries(measuredStepsPerSecond)) {
    for (const [key, rate] of Object.entries(rates)) {
      const { steps, seconds } = PRESETS[key].levels[level];
      const honest = steps / rate;
      assert.ok(
        seconds >= honest * 0.8,
        `${key} at ${level} claims ${seconds}s for ${steps} steps, but measured ` +
          `throughput needs about ${honest.toFixed(0)}s`
      );
    }
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
  // A vocabulary wide enough for top-k to bite: 120 distinct words.
  const text = Array.from({ length: 120 }, (_, i) => `word${i} `).join('').repeat(3);
  const vocab = buildVocab(text);
  const data = encode(text, vocab);
  const config = {
    ...preset('quick'),
    nLayer: 1, nHead: 2, dModel: 8, dFF: 16,
    blockSize: 8, batchSize: 1, steps: 1, weightDecay: 0,
  };
  assert.ok(vocab.size > 40, 'this corpus must be wide enough for top-k 20 to bite');
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

  // 🔴 The cap used to decide everything: at 40 epochs the built-in 4,359-character
  // name list gave every preset exactly 605 steps, so what the preset asked for was
  // irrelevant. With the ceiling at 400 and the presets at word-level step counts,
  // it no longer binds on a realistic paste.
  assert.ok(MAX_EPOCHS >= 400, `the epoch ceiling is ${MAX_EPOCHS}, low enough to decide the run`);

  const realistic = [
    ['a 12,000-character paste, the size that produced the garbage', 11939],
    ['a 30,000-character document', 30000],
  ];
  for (const [label, length] of realistic) {
    for (const key of PRESET_KEYS) {
      const { steps } = PRESETS[key].levels.word;
      assert.equal(
        plannedSteps(preset(key, 'word'), length),
        steps,
        `${key} on ${label} is capped to fewer than the ${steps} steps it asks for — ` +
          `the run-length cap is deciding how well the model learns again`
      );
    }
  }

  // A SMALL corpus is a legitimate exception, and worth stating rather than hiding:
  // the built-in name list is only about 700 word tokens, so `thorough` is capped
  // there to ~730 steps. That is 400 passes over the text, which is ample — but it
  // must still be far more than a token amount.
  for (const key of PRESET_KEYS) {
    const capped = plannedSteps(preset(key, 'word'), 4359);
    assert.ok(capped >= 250, `${key} is starved on the built-in name list (${capped} steps)`);
  }

  // At CHARACTER level the name list is 4,359 tokens and none of the presets may be
  // throttled there — that is the corpus the default demo runs on.
  for (const key of PRESET_KEYS) {
    const { steps } = PRESETS[key].levels.char;
    assert.equal(
      plannedSteps(preset(key, 'char'), 4359),
      steps,
      `${key} at character level is throttled on the name list — the default demo ` +
        `would train less than the preset asks for`
    );
  }

  // The ceiling must still exist, or a one-paragraph corpus would be read an
  // unbounded number of times and the progress estimate would be nonsense.
  assert.ok(
    plannedSteps(preset('quick', 'char'), 300) < PRESETS.quick.levels.char.steps,
    'a very short corpus must still be capped'
  );
});
