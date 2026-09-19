/**
 * 🔴 THE QUALITY GATE — the test that decides whether the output is readable.
 *
 * George trained `standard`, pressed "Sample from it", and got "thanger
 * thand-wrardeady": letter salad shaped like words. Nothing threw, nothing was
 * broken, and no existing test noticed — because the rest of the suite asserts
 * MECHANISM (gradients are right, the vocabulary round-trips, the shapes compose).
 * This file asserts OUTCOME: after its configured number of steps, every preset
 * must produce text that reads.
 *
 * It was first written against a CHARACTER-level model, and it is what caught the
 * truth that no amount of extra training fixes letter salad: at loss 0.775 the
 * output was still "thanger thand-wrardeady". That finding is why the tokenizer is
 * word-level now.
 *
 * Slow by design. `npm test` runs the fast structural checks in
 * `training.test.js`; run this with `npm run test:quality`, or
 * `PRESET=quick npm run test:quality` for one preset. It must pass before a deploy.
 *
 * ⚠️ WHAT THIS DOES NOT PROVE, stated so it is never claimed otherwise. It proves
 * the output is TEXT — ordered, readable, in the vocabulary of the corpus. It does
 * not prove the model generalises. On a corpus this size the presets largely
 * MEMORISE the document, and `copyRatio` below prints how much, so the report never
 * hides it. That is the correct outcome at this size and it is worth saying out
 * loud: the model reproduces your text, it does not understand it.
 *
 * ⚠️ CALIBRATION BOUND, stated so it is never claimed otherwise. This is measured
 * on `README.md` — 11,895 characters, 1,945 word tokens. The presets train a FIXED
 * number of steps, so a much larger paste sees each word proportionally less often
 * and its loss will be higher; about 12,000 characters, the size that produced the
 * original report, is comfortably inside the guarantee. **Scaling the step count to
 * the corpus is the obvious next improvement and has deliberately NOT been done**,
 * because it would make a large paste take proportionally longer.
 */
import { test } from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';

import {
  Trainer,
  buildVocab,
  encode,
  tokenize,
  preset,
  PRESETS,
  RECOGNISABLE_LOSS,
  COHERENCE_FLOOR,
  plannedSteps,
} from '../site/llm.js';
import { CORPORA } from '../site/corpora.js';

/**
 * 🔴 BOTH LEVELS ARE GATED, because the site ships both and they fail in OPPOSITE
 * directions. Covering only one of them is how the site got broken twice: the
 * character level was fine for a list and produced garbage on prose, and the
 * word-level fix for prose collapsed the list into reciting four names forever.
 *
 *   README.md          prose, words repeat  -> WORD level, where more training is
 *                                            strictly better
 *   the built-in names one-off items        -> CHARACTER level, where more training
 *                                            makes it RECITE instead of invent
 */
const CASES = [
  {
    label: 'README.md',
    level: 'word',
    corpus: readFileSync(new URL('../README.md', import.meta.url), 'utf8'),
  },
  { label: 'the built-in name list', level: 'char', corpus: CORPORA.names.text },
];

/** Adjacent word pairs in a text — the word-level yardstick. */
function bigramsOf(text) {
  const toks = tokenize(text, 'word').map((t) => t.trim());
  const set = new Set();
  for (let i = 1; i < toks.length; i += 1) set.add(`${toks[i - 1]} ${toks[i]}`);
  return set;
}

/** Every 5-word window in a text — used to measure how much a sample copies. */
function fivegramsOf(text) {
  const toks = tokenize(text, 'word').map((t) => t.trim());
  const set = new Set();
  for (let i = 4; i < toks.length; i += 1) set.add(toks.slice(i - 4, i + 1).join(' '));
  return set;
}

for (const c of CASES) {
  c.bigrams = c.level === 'word' ? bigramsOf(c.corpus) : null;
  c.fivegrams = fivegramsOf(c.corpus);
}

/**
 * 🔴 THERE ARE TWO METRICS BECAUSE THERE ARE TWO LEVELS, and each measures the one
 * thing that can go wrong at that level.
 *
 * WORD level — the share of a sample's adjacent word PAIRS that occur in the corpus.
 * At word level every token is already a real word, so counting words says nothing;
 * what separates text from a bag of words is the ORDER. Measured on `README.md`: an
 * under-trained run sits at 86–91%, a trained one at 100%.
 *
 * CHARACTER level — the share of a sample's 5-letters-or-longer words that occur in
 * the corpus. A character model has no words to put in the wrong order; the question
 * is whether the letter runs it emits are words at all. Alphabet soup scores 3–4%;
 * an invented-but-real name scores 30%+ because it is built from real fragments.
 *
 * The floors live in `src/llm.ts` as `COHERENCE_FLOOR`, so the bar the live page
 * judges output against is literally the same constant asserted here.
 */
function pairRatio(sample, bigrams) {
  const words = tokenize(sample, 'word').map((t) => t.trim());
  if (words.length < 2) return 0;
  let hit = 0;
  for (let i = 1; i < words.length; i += 1) {
    if (bigrams.has(`${words[i - 1]} ${words[i]}`)) hit += 1;
  }
  return hit / (words.length - 1);
}

function realWordRatio(sample, corpus) {
  const words = sample.toLowerCase().match(/[a-z]{5,}/g) ?? [];
  if (!words.length) return 0;
  const hay = corpus.toLowerCase();
  return words.filter((w) => hay.includes(w)).length / words.length;
}

/** The level's own measure, and the level's own floor. */
const coherence = (sample, c) =>
  c.level === 'word' ? pairRatio(sample, c.bigrams) : realWordRatio(sample, c.corpus);

/**
 * How much of a sample is copied word-for-word from the corpus, measured as the
 * share of its 5-word windows that appear in the corpus.
 *
 * This is REPORTED, never asserted, and at CHARACTER level it is the number that
 * matters most: over-training a list does not produce noise, it produces the corpus
 * back again. A report that showed a falling loss and called it improvement would be
 * the dishonest version — see the character-level note in `src/llm.ts`.
 */
function copyRatio(sample, fivegrams) {
  const words = tokenize(sample, 'word').map((t) => t.trim());
  if (words.length < 5) return 0;
  let hit = 0;
  let windows = 0;
  for (let i = 4; i < words.length; i += 1) {
    windows += 1;
    if (fivegrams.has(words.slice(i - 4, i + 1).join(' '))) hit += 1;
  }
  return windows ? hit / windows : 0;
}

/**
 * 🔴 A FIXED SEED, because the trainer starts from a random initialisation and a
 * gate that reports a different number every run is not a gate. The seed is
 * arbitrary — it is NOT chosen to flatter any preset.
 *
 * It earned its place: when this gate was first written, the same configuration
 * measured loss 1.310 on one unseeded run and **1.679** on the fixed seed. **A loss
 * is a sample, not a value — never set a threshold from one run.**
 */
const SEED = 20260918;

function trainAndSample(key, c) {
  const config = preset(key, c.level);
  const vocab = buildVocab(c.corpus);
  const data = encode(c.corpus, vocab);
  const trainer = new Trainer({ data, vocab, config, seed: SEED });

  // The steps the PAGE would really take — via the same `plannedSteps` the host
  // uses. Training more would be dishonest: a run capped in the browser can never
  // reach a longer one, so the cap has to be inside the thing being measured.
  const steps = plannedSteps(config, data.length);
  for (let i = 0; i < steps; i += 1) trainer.step();

  // No prompt. At word level a prompt word the corpus never contained would
  // contribute nothing; at character level the samples are meant to be new entries,
  // so priming with the corpus would defeat the point.
  const length = c.level === 'char' ? 120 : 40;
  const samples = [0, 1, 2].map((i) =>
    trainer.sample({ prompt: '', temperature: c.level === 'char' ? 0.9 : 0.7, topK: 20, length, seed: 99 + i * 7919 })
  );
  return { trainer, samples, steps, tokens: data.length, vocab, level: vocab.level };
}

const requested = process.env.PRESET;
const keys = requested ? [requested] : ['quick', 'standard', 'thorough'];

for (const c of CASES) {
  for (const key of keys) {
    test(`${key} on ${c.label} reaches readable output at ${c.level} level`, () => {
      assert.ok(PRESETS[key], `unknown preset ${key}`);
      const { trainer, samples, steps, tokens, vocab, level } = trainAndSample(key, c);
      assert.equal(level, c.level, `the corpus resolved to ${level}, not ${c.level}`);

      // 🔴 The cap is checked FIRST, because it used to make every other number here
      // meaningless: at 40 epochs each preset was cut to 605 steps on the built-in
      // corpus, so what the preset asked for did not decide anything.
      const asked = PRESETS[key].levels[c.level].steps;
      assert.equal(
        steps,
        asked,
        `${key} ran ${steps} steps, not the ${asked} it asks for — the epoch cap ` +
          `is throttling the preset, which is exactly how the presets came to be overridden`
      );

      const ratio = samples.reduce((sum, s) => sum + coherence(s, c), 0) / samples.length;
      const copied = samples.reduce((sum, s) => sum + copyRatio(s, c.fivegrams), 0) / samples.length;
      const floor = COHERENCE_FLOOR[c.level];
      const detail =
        `${key} on ${c.label} (${level} level): ${steps} steps over ${tokens} tokens ` +
        `(vocabulary ${vocab.size}), loss ${trainer.smoothedLoss.toFixed(3)} ` +
        `(must be < ${RECOGNISABLE_LOSS[level]}), ` +
        `coherence ${(ratio * 100).toFixed(0)}% (must be >= ${(floor * 100).toFixed(0)}%), ` +
        `copied from the corpus ${(copied * 100).toFixed(0)}% (reported, not asserted)\n` +
        samples.map((s) => '    * ' + s.replace(/\n/g, ' / ').slice(0, 200)).join('\n');
      console.log('\n' + detail + '\n');

      assert.ok(
        trainer.smoothedLoss < RECOGNISABLE_LOSS[level],
        `the loss is still ${trainer.smoothedLoss.toFixed(3)} — the preset is stopping too early.\n${detail}`
      );
      assert.ok(
        ratio >= floor,
        `coherence is only ${(ratio * 100).toFixed(0)}% against a floor of ` +
          `${(floor * 100).toFixed(0)}% — that is not readable output.\n${detail}`
      );
    });
  }
}
