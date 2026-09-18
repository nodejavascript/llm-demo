/**
 * What the SAMPLE CONTROLS do to the same trained model.
 *
 *   node tools/probe-defaults.mjs
 *
 * George trained on his own prose, pressed "Sample from it", and got noise. The
 * model was not the only cause: the page's sample defaults are `prompt = ''` and
 * `topK = 0`, and a top-k of zero means NO truncation — the sampler draws from
 * all 70-odd characters of the vocabulary, so low-probability junk is picked
 * constantly and there is no context to continue from.
 *
 * This trains one model on real prose and samples it several ways, so the
 * contribution of the controls can be seen separately from the contribution of
 * the model. `README.md` is used as the corpus because it is genuine technical
 * prose of about the size George pasted (11,939 characters).
 *
 * Not part of the test suite: it prints a judgement, not a pass or a fail.
 */
import { readFileSync } from 'node:fs';

import { Trainer, preset, buildVocab, encode } from '../site/llm.js';

const TEXT = readFileSync(new URL('../README.md', import.meta.url), 'utf8');

const config = preset('standard');
const vocab = buildVocab(TEXT);
const data = encode(TEXT, vocab);

console.log(
  `\ncorpus: ${TEXT.length} characters, vocabulary of ${vocab.size} characters`
);
console.log(
  `preset: ${config.nLayer} layer(s), dModel ${config.dModel}, context ${config.blockSize}, ${config.steps} steps\n`
);

const trainer = new Trainer({ data, vocab, config });
console.log('parameters:', trainer.paramCount.toLocaleString());

const started = Date.now();
for (let i = 0; i < config.steps; i += 1) trainer.step();
console.log(
  `trained in ${((Date.now() - started) / 1000).toFixed(1)}s, loss ${trainer.smoothedLoss.toFixed(3)}` +
    ` (guessing among ${vocab.size} characters starts near ${Math.log(vocab.size).toFixed(2)})\n`
);

/** A real line out of the corpus, to prime with. */
const seed = TEXT.split('\n').find((l) => l.trim().length > 30)?.trim().slice(0, 24) ?? 'the';

const runs = [
  ['A · page defaults — no prompt, temperature 0.80, top-k 0 (no truncation)', { prompt: '', temperature: 0.8, topK: 0, length: 200 }],
  ['B · same, but top-k 40', { prompt: '', temperature: 0.8, topK: 40, length: 200 }],
  ['C · primed with a real line, temperature 0.80, top-k 40', { prompt: seed, temperature: 0.8, topK: 40, length: 200 }],
  ['D · primed, cooler — temperature 0.50, top-k 40', { prompt: seed, temperature: 0.5, topK: 40, length: 200 }],
];

for (const [label, options] of runs) {
  console.log(`--- ${label} ---`);
  for (let i = 0; i < 3; i += 1) {
    const text = trainer.sample({ ...options, seed: 4242 + i * 7919 });
    console.log('  * ' + text.replace(/\n/g, ' / '));
  }
  console.log();
}

console.log(`(the seed line used for C and D was: ${JSON.stringify(seed)})\n`);
