/**
 * bench.js — measures what each preset actually costs, on this machine.
 *
 * The preset step counts in site/llm.js are set from this measurement rather
 * than guessed, so the labels on the page ("seconds", "about a minute") are
 * true. Run it again on a different machine before trusting those numbers.
 *
 *   node tools/bench.js
 *   node tools/bench.js --steps 200 --preset quick
 */

import { Trainer, buildVocab, encode, preset, PRESETS, parameterCount } from '../site/llm.js';
import { CORPORA } from '../site/corpora.js';

const args = process.argv.slice(2);
const getArg = (name, fallback) => {
  const i = args.indexOf(name);
  return i === -1 ? fallback : args[i + 1];
};

const onlyPreset = getArg('--preset', null);
const corpusKey = getArg('--corpus', 'names');
const stepOverride = Number(getArg('--steps', 0)) || 0;

const text = CORPORA[corpusKey].text;
const vocab = buildVocab(text);
const data = encode(text, vocab);

console.log(`corpus "${corpusKey}": ${text.length} characters, ${vocab.size} distinct characters\n`);

const rows = [];
for (const key of Object.keys(PRESETS)) {
  if (onlyPreset && key !== onlyPreset) continue;
  const config = { ...preset(key) };
  if (stepOverride) config.steps = stepOverride;
  const t = new Trainer({ data, vocab, config, seed: 20260917 });

  // warm up (the JIT needs a few steps before the timing means anything)
  for (let i = 0; i < 10; i++) t.step();

  const started = process.hrtime.bigint();
  for (let i = 0; i < config.steps; i++) t.step();
  const elapsed = Number(process.hrtime.bigint() - started) / 1e9;

  const early = t.lossHistory.slice(10, 20).reduce((a, b) => a + b, 0) / 10;
  const late = t.lossHistory.slice(-20).reduce((a, b) => a + b, 0) / 20;
  const sample = t.sample({ length: 160, temperature: 0.7, topK: 0, seed: 5 });

  rows.push({
    preset: key,
    params: t.paramCount,
    tokens: t.tokensSeen,
    seconds: elapsed,
    stepsPerSecond: config.steps / elapsed,
    tokensPerSecond: t.tokensSeen / elapsed,
    lossEarly: early,
    lossLate: late,
    sample,
  });
}

for (const r of rows) {
  console.log('='.repeat(72));
  console.log(`${r.preset}: ${r.params.toLocaleString()} parameters, ${r.tokens.toLocaleString()} tokens in ${r.seconds.toFixed(1)}s`);
  console.log(`  ${r.stepsPerSecond.toFixed(1)} steps/s · ${Math.round(r.tokensPerSecond).toLocaleString()} tokens/s`);
  console.log(`  loss ${r.lossEarly.toFixed(3)} → ${r.lossLate.toFixed(3)}`);
  console.log('  sample:');
  console.log(
    r.sample
      .split('\n')
      .map((l) => `    | ${l}`)
      .join('\n')
  );
}

console.log('='.repeat(72));
console.log('parameter count by preset and vocabulary size (names corpus):');
for (const key of Object.keys(PRESETS)) {
  console.log(`  ${key.padEnd(9)} ${parameterCount(preset(key), vocab.size).toLocaleString()}`);
}
