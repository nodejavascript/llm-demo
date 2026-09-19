import { test } from 'node:test';
import assert from 'node:assert/strict';

import {
  Trainer,
  buildVocab,
  encode,
  decode,
  tokenize,
  detectLevel,
  lineStartToken,
  parameterCount,
  preset,
  PRESETS,
  mulberry32,
} from '../site/llm.js';

const SENTENCE = 'the quick brown fox jumps over the lazy dog. ';
const CORPUS = SENTENCE.repeat(30);

function makeTrainer(overrides = {}, text = CORPUS) {
  const vocab = buildVocab(text);
  const data = encode(text, vocab);
  const config = {
    ...preset('quick'),
    nLayer: 2, nHead: 2, dModel: 8, dFF: 16,
    blockSize: 6, batchSize: 1, steps: 1, weightDecay: 0,
    ...overrides,
  };
  return new Trainer({ data, vocab, config, seed: 7 });
}

/* ------------------------------------------------------------------ *
 * Tokenizer
 * ------------------------------------------------------------------ */

test('vocabulary round-trips text losslessly', () => {
  const text = 'Hello, world!\nHello again — 42.';
  const vocab = buildVocab(text);
  const ids = encode(text, vocab);
  assert.equal(decode(ids, vocab), text);
  assert.equal(new Set(vocab.chars).size, vocab.chars.length, 'no duplicate characters');
});

test('words beyond the vocabulary cap fold into the unknown bucket', () => {
  // 200 distinct words, then the cap forces the rarest into one bucket.
  //
  // 🔴 THE LEVEL IS NAMED HERE, AND THAT IS NOT PEDANTRY. A corpus of 200 words that
  // each occur ONCE is exactly the shape `detectLevel` classifies as CHARACTER level,
  // so the bare `buildVocab(text, 64)` no longer built a word vocabulary at all — it
  // built a 15-token character one, the cap never bound, and this test failed the
  // moment the level became automatic. The cap is a WORD-level concern, so the word
  // level is what it asks for.
  const many = Array.from({ length: 200 }, (_, i) => `w${i} `).join('');
  const text = 'common '.repeat(4) + many + 'common '.repeat(4);
  const vocab = buildVocab(text, 64, 'word');
  assert.equal(vocab.size, 64);
  assert.equal(vocab.unkIndex, 63);
  const ids = encode(text, vocab);
  assert.equal(ids.length, tokenize(text, 'word').length);
  // a word never seen at all still encodes, into the unknown bucket
  const fresh = encode('zzzz ', vocab);
  assert.equal(fresh[0], vocab.unkIndex);
  // and the unknown bucket decodes to NOTHING, so it never prints a marker
  assert.equal(vocab.chars[vocab.unkIndex], '');
});

test('detectLevel sends a list of one-off items to the characters and prose to words', () => {
  // 🔴 THE MEASUREMENT THE WHOLE TWO-LEVEL DESIGN RESTS ON. Get this wrong and the
  // site breaks in one direction or the other: a list tokenised by word can only
  // recite itself, and prose tokenised by character comes out as letter salad.
  // Measured on the real corpora — the 653-name list scores 1.00 distinct words,
  // the dialogue 0.34 — and the threshold sits between them at 0.8.
  const list = Array.from({ length: 300 }, (_, i) => `entry${i} `).join('');
  assert.equal(detectLevel(list), 'char');

  const prose = 'the cat sat on the mat and the cat sat on the other mat again '.repeat(20);
  assert.equal(detectLevel(prose), 'word');

  // Too little text to judge a ratio from, so it stays at word level rather than
  // guessing from a handful of tokens.
  assert.equal(detectLevel('a b c d e f g h'), 'word');

  // And the vocabulary records which level built it, because `encode` and `decode`
  // read it back off the vocab rather than being told again.
  assert.equal(buildVocab(prose).level, 'word');
  assert.equal(buildVocab(list).level, 'char');
});

test('a corpus with more than one code point per grapheme encodes correctly', () => {
  const text = 'a😀 b😀 c😀';
  const vocab = buildVocab(text);
  const ids = encode(text, vocab);
  assert.equal(ids.length, 3);
  assert.equal(decode(ids, vocab), text);
});

test('lineStartToken finds the word lines begin with', () => {
  const text = 'apple pie\napple tart\napple cake\n';
  const vocab = buildVocab(text);
  const data = encode(text, vocab);
  assert.equal(vocab.chars[lineStartToken(data, vocab)], 'apple ');
});

/* ------------------------------------------------------------------ *
 * The gradient check — the test that makes the backprop trustworthy
 * ------------------------------------------------------------------ */

/**
 * Numerical gradient of the mean loss with respect to one weight.
 *
 * The weights are float32, so the loss itself carries rounding noise of about
 * 1e-7 — which means a finite difference can only measure gradients that are
 * large enough for the perturbation to lift the loss clear of that noise. The
 * check therefore targets, in every parameter array, the entry with the
 * largest analytic gradient, and uses an eps scaled to the weight. Every part
 * of the network (attention, MLP, both layer norms, the head, the embeddings)
 * has such an entry, and a wrong backward pass in any of them shows up here.
 */
function numericGrad(t, p, i) {
  const eps = 1e-3 * Math.max(1, Math.abs(p.w[i]));
  const orig = p.w[i];
  p.w[i] = orig + eps;
  const plus = t.lossAt(0);
  p.w[i] = orig - eps;
  const minus = t.lossAt(0);
  p.w[i] = orig;
  return (plus - minus) / (2 * eps);
}

function relError(a, b) {
  return Math.abs(a - b) / Math.max(1e-12, Math.abs(a) + Math.abs(b));
}

test('analytical gradients match numerical gradients', () => {
  const t = makeTrainer();
  const M = t.gradsAt(0, 1);
  assert.equal(M, 6, 'the check runs over one full window');

  const analytic = t.params.map((p) => Float32Array.from(p.g));
  const checked = [];
  let worst = 0;
  let worstName = '';

  for (let pi = 0; pi < t.params.length; pi++) {
    const p = t.params[pi];
    let best = 0;
    for (let i = 1; i < p.w.length; i++) {
      if (Math.abs(analytic[pi][i]) > Math.abs(analytic[pi][best])) best = i;
    }
    const a = analytic[pi][best];
    assert.ok(Math.abs(a) > 1e-6, `${p.name} has a gradient worth checking (${a})`);
    const numeric = numericGrad(t, p, best);
    const err = relError(numeric, a);
    checked.push(`${p.name}[${best}]`);
    if (err > worst) {
      worst = err;
      worstName = `${p.name}[${best}] numeric=${numeric.toExponential(3)} analytic=${a.toExponential(3)}`;
    }
  }

  assert.ok(checked.length >= t.params.length, `checked ${checked.length} parameters`);
  assert.ok(worst < 0.02, `worst relative gradient error ${worst.toFixed(5)} at ${worstName}`);
});

test('the gradient check has power — a corrupted gradient is caught', () => {
  const t = makeTrainer();
  t.gradsAt(0, 1);
  const p = t.head;
  let best = 0;
  for (let i = 1; i < p.w.length; i++) if (Math.abs(p.g[i]) > Math.abs(p.g[best])) best = i;
  const analytic = p.g[best];
  const numeric = numericGrad(t, p, best);
  assert.ok(relError(numeric, analytic) < 0.02, 'the uncorrupted gradient matches');
  assert.ok(relError(numeric, analytic * 1.5) > 0.1, 'a 50% wrong gradient is detected');
  assert.ok(relError(numeric, -analytic) > 0.5, 'a sign error is detected');
});

/* ------------------------------------------------------------------ *
 * Training
 * ------------------------------------------------------------------ */

test('training reduces the loss', () => {
  const t = makeTrainer({ nLayer: 1, dModel: 16, dFF: 32, blockSize: 16, batchSize: 8, lr: 5e-3 });
  for (let i = 0; i < 40; i++) t.step();
  const early = t.lossHistory.slice(0, 10).reduce((a, b) => a + b, 0) / 10;
  for (let i = 0; i < 200; i++) t.step();
  const late = t.lossHistory.slice(-40).reduce((a, b) => a + b, 0) / 40;

  assert.ok(late < early * 0.6, `loss fell from ${early.toFixed(3)} to ${late.toFixed(3)}`);
  assert.ok(late < 2.0, `final loss is ${late.toFixed(3)}`);
  assert.equal(t.stepCount, 240);
  assert.ok(t.tokensSeen > 0);
});

test('training is deterministic for a given seed', () => {
  const a = makeTrainer({ nLayer: 1, dModel: 16, dFF: 32, blockSize: 16, batchSize: 4, lr: 5e-3 });
  const b = makeTrainer({ nLayer: 1, dModel: 16, dFF: 32, blockSize: 16, batchSize: 4, lr: 5e-3 });
  for (let i = 0; i < 15; i++) { a.step(); b.step(); }
  assert.deepEqual(a.lossHistory, b.lossHistory);
  assert.equal(a.sample({ length: 30, seed: 99 }), b.sample({ length: 30, seed: 99 }));
});

test('a different seed gives a different model', () => {
  const vocab = buildVocab(CORPUS);
  const data = encode(CORPUS, vocab);
  const config = { ...preset('quick'), nLayer: 1, dModel: 8, dFF: 16, blockSize: 8, batchSize: 2 };
  const a = new Trainer({ data, vocab, config, seed: 1 });
  const b = new Trainer({ data, vocab, config, seed: 2 });
  assert.notEqual(a.wte.w[0], b.wte.w[0]);
});

/* ------------------------------------------------------------------ *
 * Generation
 * ------------------------------------------------------------------ */

test('samples come back inside the requested length and vocabulary', () => {
  const t = makeTrainer({ nLayer: 1, dModel: 16, dFF: 32, blockSize: 16, batchSize: 4, lr: 5e-3 });
  for (let i = 0; i < 20; i++) t.step();
  const out = t.sample({ length: 80, temperature: 0.8 });
  // `length` counts TOKENS now, not characters, so the string can be much longer
  // than 80 — the invariant is the number of words drawn.
  const words = tokenize(out);
  assert.ok(words.length > 0 && words.length <= 80, `words ${words.length}`);
  for (const word of words) {
    assert.ok(t.vocab.chars.includes(word), `word ${JSON.stringify(word)} is in the vocabulary`);
  }
});

test('a prompt is honoured as the start of the context', () => {
  const t = makeTrainer({ nLayer: 1, dModel: 16, dFF: 32, blockSize: 16, batchSize: 4 });
  const out = t.sample({ prompt: 'the ', length: 20, temperature: 0.5 });
  assert.equal(typeof out, 'string');
  assert.ok(out.length > 0);
});

test('temperature 0 is greedy and repeats under the same model', () => {
  const t = makeTrainer({ nLayer: 1, dModel: 16, dFF: 32, blockSize: 16, batchSize: 4 });
  for (let i = 0; i < 10; i++) t.step();
  const a = t.sample({ length: 40, temperature: 0, seed: 5 });
  const b = t.sample({ length: 40, temperature: 0, seed: 5 });
  assert.equal(a, b);
});

test('top-k narrows the sample and is repeatable', () => {
  // A barely trained model is close to uniform, so restricting to the single
  // most likely character must produce far less variety than sampling freely.
  const t = makeTrainer({ nLayer: 1, dModel: 16, dFF: 32, blockSize: 16, batchSize: 4 });
  const distinct = (s) => new Set([...s]).size;
  const narrow = t.sample({ length: 200, temperature: 1, topK: 1, seed: 11 });
  const wide = t.sample({ length: 200, temperature: 1, topK: 0, seed: 11 });
  assert.ok(distinct(narrow) < distinct(wide), `top-1 gave ${distinct(narrow)} characters, unrestricted gave ${distinct(wide)}`);
  assert.equal(narrow, t.sample({ length: 200, temperature: 1, topK: 1, seed: 11 }));
});

/* ------------------------------------------------------------------ *
 * Shapes, exports, seeds
 * ------------------------------------------------------------------ */

test('parameterCount agrees with the built model', () => {
  const vocab = buildVocab(CORPUS);
  for (const key of Object.keys(PRESETS)) {
    const config = preset(key);
    const t = new Trainer({ data: encode(CORPUS, vocab), vocab, config, seed: 3 });
    assert.equal(parameterCount(config, vocab.size), t.paramCount, `${key} preset`);
  }
});

test('exportWeights and exportReport describe the model', () => {
  const t = makeTrainer({ nLayer: 1, dModel: 16, dFF: 32, blockSize: 16, batchSize: 4 });
  for (let i = 0; i < 5; i++) t.step();

  const w = t.exportWeights();
  assert.equal(w.params.wte.length, t.vocabSize * t.C);
  assert.equal(w.vocab.length, t.vocabSize);
  assert.equal(w.config.blockSize, t.T);

  const r = t.exportReport({ samples: ['x'] });
  assert.equal(r.model.parameters, t.paramCount);
  assert.equal(r.model.optimizer, 'AdamW');
  assert.equal(r.training.steps, 5);
  assert.ok(r.training.curve.length >= 2);
  assert.equal(r.samples[0], 'x');
  assert.match(r.site, /^https:\/\//);
});

test('a corpus shorter than the context shrinks the context instead of failing', () => {
  const text = 'one two three';
  const vocab = buildVocab(text);
  const data = encode(text, vocab);
  const t = new Trainer({
    data,
    vocab,
    config: { ...preset('quick'), blockSize: 64 },
  });
  assert.ok(t.T <= data.length - 1);
  t.step();
  assert.equal(t.stepCount, 1);
});

test('an empty corpus is refused', () => {
  const vocab = buildVocab('abc');
  assert.throws(() => new Trainer({ data: new Int32Array(0), vocab, config: preset('quick') }));
});

test('mulberry32 is deterministic and in range', () => {
  const a = mulberry32(42);
  const b = mulberry32(42);
  for (let i = 0; i < 100; i++) {
    const v = a();
    assert.equal(v, b());
    assert.ok(v >= 0 && v < 1);
  }
});

test('every preset trains a step without error, at both levels', () => {
  for (const key of Object.keys(PRESETS)) {
    for (const level of ['word', 'char']) {
      const text = CORPUS.repeat(4);
      const vocab = buildVocab(text, 2000, level);
      const t = new Trainer({
        data: encode(text, vocab),
        vocab,
        config: { ...preset(key, level), steps: 1 },
      });
      const { loss } = t.step();
      assert.ok(Number.isFinite(loss) && loss > 0, `${key} at ${level} level, loss ${loss}`);
    }
  }
});
