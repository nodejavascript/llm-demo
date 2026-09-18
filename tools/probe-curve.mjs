/**
 * How much training does a shape need before the output is recognisable?
 *
 *   node tools/probe-curve.mjs                  # every preset
 *   PRESET=quick node tools/probe-curve.mjs     # just one
 *   CORPUS=big.txt node tools/probe-curve.mjs   # a different document
 *
 * This is the rig the step counts were set from. It trains one model per preset
 * and, at each mark, prints the smoothed loss beside the share of the sample's
 * 5+ character words that actually occur in the corpus. That second number is the
 * one that matters: it is the difference between "the loss fell" and "the text is
 * no longer garbage".
 *
 * Measured 18 Sep 2026 on README.md (8,389 characters), which is where the
 * boundary in `RECOGNISABLE_LOSS` comes from:
 *
 *     loss 2.47 -> 3%    loss 1.65 -> 13%    loss 0.91 -> 31%
 *     loss 2.07 -> 4%    loss 1.36 -> 23%
 *     loss 1.94 -> 4%
 *
 * Not part of the test suite: it prints a judgement, not a pass or a fail. The
 * assertion lives in `test/quality.test.js`.
 */
import { readFileSync } from 'node:fs';

import { Trainer, preset, buildVocab, encode } from '../site/llm.js';

const CORPUS = readFileSync(new URL(`../${process.env.CORPUS ?? 'README.md'}`, import.meta.url), 'utf8');

/** The share of the sample's 5+ character words that really occur in the corpus. */
function realWordRatio(sample, corpus) {
  const words = sample.toLowerCase().match(/[a-z]{5,}/g) ?? [];
  if (!words.length) return 0;
  const hay = corpus.toLowerCase();
  return words.filter((w) => hay.includes(w)).length / words.length;
}

const marks = (process.env.MARKS ?? '250,500,1000,2000,4000,8000')
  .split(',')
  .map(Number)
  .filter((n) => Number.isFinite(n) && n > 0);

const keys = process.env.PRESET ? [process.env.PRESET] : ['quick', 'standard', 'thorough'];

console.log(`\ncorpus: ${CORPUS.length} characters, vocabulary of ${buildVocab(CORPUS).size}`);

for (const key of keys) {
  const config = preset(key);
  const vocab = buildVocab(CORPUS);
  const data = encode(CORPUS, vocab);
  const trainer = new Trainer({ data, vocab, config });
  const budget = Math.max(...marks.filter((m) => m <= config.steps * 3), config.steps);

  console.log(
    `\n########## ${key} · ${config.nLayer}L d${config.dModel} ctx${config.blockSize} · ` +
      `${trainer.paramCount.toLocaleString()} params · configured ${config.steps} steps`
  );

  const started = Date.now();
  for (let i = 1; i <= budget; i += 1) {
    trainer.step();
    if (marks.includes(i) || i === config.steps) {
      const secs = (Date.now() - started) / 1000;
      let real = 0;
      let words = 0;
      for (let k = 0; k < 3; k += 1) {
        const out = trainer.sample({
          prompt: 'A small GPT',
          temperature: 0.7,
          topK: 40,
          length: 170,
          seed: 99 + k * 7919,
        });
        const sampleWords = out.toLowerCase().match(/[a-z]{5,}/g) ?? [];
        words += sampleWords.length;
        const hay = CORPUS.toLowerCase();
        real += sampleWords.filter((w) => hay.includes(w)).length;
      }
      const ratio = words ? (real / words) * 100 : 0;
      const flag = ratio >= 12 ? 'ok  ' : 'NOISE';
      console.log(
        `  ${flag} step ${String(i).padStart(5)}  loss ${trainer.smoothedLoss.toFixed(3)}  ` +
          `${secs.toFixed(0).padStart(4)}s  real words ${ratio.toFixed(0)}%`
      );
    }
  }
}
