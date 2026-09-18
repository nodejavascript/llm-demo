/**
 * llm.ts — a small GPT, written from scratch in TypeScript.
 *
 * No runtime dependencies. `tsc` compiles this file to one plain JavaScript
 * module the browser loads directly: no bundler, no framework, no server, no API
 * key. Everything a language model needs is here — a tokenizer, token and
 * position embeddings, causal multi-head self-attention, a feed-forward block,
 * layer normalisation, residual connections, softmax cross-entropy,
 * hand-derived backpropagation and AdamW, plus sampling with temperature and
 * top-k.
 *
 * It is deliberately small. `quick` is about thirteen thousand parameters and
 * trains on one CPU core in seconds; a frontier model is roughly a hundred
 * thousand times larger again and is trained on far more text. This
 * demonstrates the mechanism, not the scale.
 *
 * Deterministic: every random choice comes from a seeded generator, so a given
 * corpus, config and seed always produce the same model.
 */

/* ------------------------------------------------------------------ *
 * Types
 * ------------------------------------------------------------------ */

/** One model size. `steps` and `seconds` come from measurement, never a guess. */
export interface ModelConfig {
  key: string;
  label: string;
  blurb: string;
  /** Seconds the FIRST run in a fresh tab takes — what a visitor actually meets. */
  seconds: number;
  nLayer: number;
  nHead: number;
  dModel: number;
  dFF: number;
  blockSize: number;
  batchSize: number;
  lr: number;
  steps: number;
  weightDecay: number;
}

export interface Vocab {
  chars: string[];
  stoi: Map<string, number>;
  size: number;
  /** Index of the "unknown" bucket, or -1 when nothing was folded into it. */
  unkIndex: number;
}

export interface SampleOptions {
  prompt?: string;
  length?: number;
  temperature?: number;
  topK?: number;
  seed?: number | null;
}

export interface StepResult {
  loss: number;
  tokens: number;
}

export interface ExportedWeights {
  name: string;
  architecture: string;
  config: ModelConfig;
  vocab: string[];
  params: Record<string, number[]>;
}

export interface TrainingReport {
  site: string;
  generated_at: string;
  model: {
    parameters: number;
    layers: number;
    heads: number;
    width: number;
    feed_forward: number;
    context: number;
    batch_size: number;
    learning_rate: number;
    optimizer: string;
    weight_decay: number;
    tokenizer: string;
    vocabulary_size: number;
  };
  training: {
    steps: number;
    tokens_seen: number;
    corpus_characters: number;
    final_loss: number | null;
    smoothed_loss: number | null;
    curve: number[];
  };
  [extra: string]: unknown;
}

/* ------------------------------------------------------------------ *
 * Random numbers (seeded, so runs are reproducible and testable)
 * ------------------------------------------------------------------ */

export function mulberry32(seed: number): () => number {
  let a = seed >>> 0;
  return function next(): number {
    a = (a + 0x6d2b79f5) | 0;
    let t = Math.imul(a ^ (a >>> 15), 1 | a);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

function gaussian(rand: () => number): number {
  let u = 0;
  while (u === 0) u = rand();
  return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * rand());
}

/* ------------------------------------------------------------------ *
 * Tokenizer — character level, built from the corpus itself
 * ------------------------------------------------------------------ */

function codePointCount(text: string): number {
  let n = 0;
  for (let i = 0; i < text.length; i++) {
    const c = text.charCodeAt(i);
    if (c >= 0xd800 && c <= 0xdbff && i + 1 < text.length) i++;
    n++;
  }
  return n;
}

function pickUnusedChar(text: string, taken: Set<string>): string {
  for (let cp = 0xe000; cp <= 0xf8ff; cp++) {
    const ch = String.fromCodePoint(cp);
    if (!taken.has(ch) && !text.includes(ch)) return ch;
  }
  return '\u0000';
}

/**
 * Build the character vocabulary from the corpus. The most frequent characters
 * are kept; anything beyond `maxVocab` is folded into one "unknown" bucket, so
 * a corpus full of emoji cannot blow the model up.
 */
export function buildVocab(text: string, maxVocab = 128): Vocab {
  const freq = new Map<string, number>();
  for (const ch of text) freq.set(ch, (freq.get(ch) ?? 0) + 1);
  let chars = [...freq.keys()].sort(
    (a, b) => (freq.get(b) ?? 0) - (freq.get(a) ?? 0) || (a < b ? -1 : a > b ? 1 : 0)
  );
  let unknown: string | null = null;
  if (chars.length > maxVocab) {
    const keep = new Set(chars.slice(0, maxVocab - 1));
    unknown = pickUnusedChar(text, keep);
    chars = chars.slice(0, maxVocab - 1);
    chars.push(unknown);
  }
  const stoi = new Map<string, number>();
  chars.forEach((ch, i) => stoi.set(ch, i));
  return { chars, stoi, size: chars.length, unkIndex: unknown ? chars.length - 1 : -1 };
}

export function encode(text: string, vocab: Vocab): Int32Array {
  const out = new Int32Array(codePointCount(text));
  let i = 0;
  for (const ch of text) {
    const id = vocab.stoi.get(ch);
    out[i++] = id === undefined ? vocab.unkIndex : id;
  }
  return out;
}

export function decode(ids: ArrayLike<number>, vocab: Vocab): string {
  const out: string[] = [];
  for (let i = 0; i < ids.length; i++) out.push(vocab.chars[ids[i]] ?? '');
  return out.join('');
}

/** The character a line most often starts with — the seed for an empty prompt. */
export function lineStartToken(data: Int32Array, vocab: Vocab): number {
  const counts = new Int32Array(vocab.size);
  let prev = -1;
  for (let i = 0; i < data.length; i++) {
    if (prev === -1 || vocab.chars[prev] === '\n') counts[data[i]]++;
    prev = data[i];
  }
  let best = 0;
  let bestN = 0;
  for (let i = 0; i < counts.length; i++) {
    if (counts[i] > bestN) {
      bestN = counts[i];
      best = i;
    }
  }
  return best;
}

/* ------------------------------------------------------------------ *
 * Model shapes
 * ------------------------------------------------------------------ */

/**
 * 🔴 STEPS ARE CHOSEN FROM THE LOSS, NOT FROM THE CLOCK — and they were once
 * chosen from the clock, which is why the demo produced garbage.
 *
 * Measured 18 Sep 2026, `README.md` as the corpus (8,389 characters, vocabulary
 * of 84), sampling primed from a real line at temperature 0.7 with top-k 40. The
 * relationship between the smoothed loss and whether the output contains real
 * words is sharp and monotonic:
 *
 *     loss 2.47  ->  3% of the words in the sample occur in the corpus
 *     loss 2.07  ->  4%
 *     loss 1.94  ->  4%
 *     loss 1.65  -> 13%
 *     loss 1.36  -> 23%
 *     loss 0.91  -> 31%
 *
 * **Below about 1.6 the output is recognisable; above about 1.9 it is noise.**
 * `RECOGNISABLE_LOSS` is that boundary and `test/quality.test.js` asserts every
 * preset clears it. The step counts below are the measured minimum each shape
 * needs to get there. They were previously 600 / 250 / 300, which left all three
 * presets between 1.9 and 2.5 — squarely in the noise, which is exactly what a
 * visitor saw.
 *
 * 🔴 A LOSS IS A SAMPLE, NOT A VALUE. The trainer starts from a random
 * initialisation, so the same shape at the same step count lands on a different
 * loss on a different run. `thorough` was first given 800 steps because ONE run
 * of that shape measured 1.310 — and a later run of the identical configuration
 * measured **1.679**, over the line, which `test/quality.test.js` caught and
 * failed on. The lesson is not to set a threshold from a single run: a preset
 * needs MARGIN, not a passing sample. The quality gate now trains with a fixed
 * seed so it is reproducible, and the presets are chosen to clear 1.6 with room
 * rather than to sit just under it.
 *
 * That is why `thorough` is a 2-layer model trained twice as long rather than a
 * 3-layer one trained briefly. Measured on the same corpus: 3 layers costs
 * 389 ms a step and needs far more than 800 of them to converge, while the
 * 2-layer shape reaches loss **0.601** in 3,000 steps and 254 s — better text,
 * and less than half the wait.
 *
 * And a preset has to clear the threshold with MARGIN, not by a hair. `quick` was
 * first given 3,000 steps, where it measured 1.476 on one seed and **1.567** on
 * the fixed seed the gate uses — passing by 0.033. That is not a margin, it is a
 * coin toss, so `quick` now trains 4,000 steps and measures about **1.36**.
 *
 * `seconds` is the time the FIRST run in a fresh tab takes — the one a visitor
 * actually meets — because the browser has to compile the loops before it
 * settles. Measured on a desktop CPU: about 87 steps/s for `quick`, 11 for
 * `standard`, 2.6 for `thorough` (389 ms a step). The page reports the rate it
 * is really achieving and takes its estimate from that rate, so a slower machine
 * says so rather than quietly taking longer. Any run can be stopped early and
 * tested as it stands — but stopping early is what produces the nonsense, so the
 * page now says that too.
 */

/**
 * The smoothed loss below which the output stops being noise.
 *
 * Not a taste judgement — it is where real words start appearing in the samples,
 * measured across both model shapes (table above). Above it the model is still
 * learning character frequencies and cannot form words; below it, it can.
 */
export const RECOGNISABLE_LOSS = 1.6;

export const PRESETS: Record<'quick' | 'standard' | 'thorough', ModelConfig> = {
  quick: {
    key: 'quick', label: 'Quick', blurb: 'under a minute — the smallest, trained enough to form words',
    seconds: 50,
    nLayer: 1, nHead: 2, dModel: 32, dFF: 64,
    blockSize: 24, batchSize: 12, lr: 3e-3, steps: 4000, weightDecay: 0.01,
  },
  standard: {
    key: 'standard', label: 'Standard', blurb: 'about two minutes — learns distinctly more of the language',
    seconds: 135,
    nLayer: 2, nHead: 2, dModel: 64, dFF: 128,
    blockSize: 32, batchSize: 12, lr: 2e-3, steps: 1500, weightDecay: 0.01,
  },
  thorough: {
    key: 'thorough', label: 'Thorough', blurb: 'about four minutes — Standard, trained twice as long',
    seconds: 255,
    nLayer: 2, nHead: 2, dModel: 64, dFF: 128,
    blockSize: 32, batchSize: 12, lr: 2e-3, steps: 3000, weightDecay: 0.01,
  },
};

export type PresetKey = keyof typeof PRESETS;

export function preset(name: string): ModelConfig {
  const found = PRESETS[name as PresetKey];
  return { ...(found ?? PRESETS.quick) };
}

/**
 * The ceiling on how many times a run may read its corpus.
 *
 * 🔴 THIS WAS 40, AND IT WAS THE SECOND HALF OF THE GARBAGE BUG. A preset's
 * `steps` were quietly overridden by this cap, because the cap is computed from
 * the CORPUS rather than from the preset:
 *
 *     steps = min(config.steps, (dataLength - 1) * 40 / (batchSize * blockSize))
 *
 * On the 4,359-character built-in name list that is **605 steps for every
 * preset** — so raising `quick` from 600 to 3,000 would have changed nothing at
 * all. On a 12,000-character paste, `standard` would have been cut from 1,500 to
 * about 1,250. The cap was there so the progress bar could not promise an absurd
 * number of passes over a tiny text, which is a real concern, but 40 was low
 * enough to be the thing that decided how well the model learned — and the loss
 * has to fall past `RECOGNISABLE_LOSS` for the output to be words at all.
 *
 * So the ceiling is now high enough not to bind on any realistic paste, while
 * still keeping a one-paragraph corpus from being read ten thousand times.
 */
export const MAX_EPOCHS = 400;

/**
 * How many steps a run will really take, which is what the progress bar, the
 * time estimate, and the quality test must all agree on.
 *
 * Exported so that `test/quality.test.js` trains the run the PAGE would train
 * rather than a longer one the page can never reach — a test that bypasses this
 * would have passed while the cap was still throttling everything.
 */
export function plannedSteps(config: ModelConfig, dataLength: number): number {
  const perStep = Math.max(1, config.batchSize * config.blockSize);
  const ceiling = Math.max(1, Math.floor(((dataLength - 1) * MAX_EPOCHS) / perStep));
  return Math.max(1, Math.min(config.steps, ceiling));
}

export function parameterCount(config: ModelConfig, vocabSize: number): number {
  const { nLayer, dModel: C, dFF: F, blockSize: T } = config;
  const perLayer = 3 * C * C + 3 * C + C * C + C + C * F + F + F * C + C + 4 * C;
  return vocabSize * C + T * C + nLayer * perLayer + 2 * C + C * vocabSize + vocabSize;
}

/* ------------------------------------------------------------------ *
 * Kernels
 * ------------------------------------------------------------------ */

function fill(arr: Float32Array, value: number, n: number = arr.length): void {
  for (let i = 0; i < n; i++) arr[i] = value;
}

function addInto(dst: Float32Array, src: Float32Array, n: number): void {
  for (let i = 0; i < n; i++) dst[i] += src[i];
}

/** y[M,N] = x[M,K] · W[K,N] + b[N] */
function linearFwd(
  x: Float32Array, M: number, K: number, W: Float32Array, N: number, b: Float32Array | null, y: Float32Array
): void {
  for (let i = 0; i < M; i++) {
    const yo = i * N;
    const xo = i * K;
    if (b) for (let j = 0; j < N; j++) y[yo + j] = b[j];
    else for (let j = 0; j < N; j++) y[yo + j] = 0;
    for (let k = 0; k < K; k++) {
      const xv = x[xo + k];
      if (xv === 0) continue;
      const wo = k * N;
      for (let j = 0; j < N; j++) y[yo + j] += xv * W[wo + j];
    }
  }
}

/** Accumulates dW += xᵀ·dy, db += Σᵢ dy, and optionally dx += dy·Wᵀ. */
function linearBwd(
  dy: Float32Array, x: Float32Array, M: number, K: number,
  W: Float32Array, N: number,
  dW: Float32Array, db: Float32Array | null, dx: Float32Array | null
): void {
  for (let i = 0; i < M; i++) {
    const yo = i * N;
    const xo = i * K;
    for (let k = 0; k < K; k++) {
      const xv = x[xo + k];
      const wo = k * N;
      let acc = 0;
      for (let j = 0; j < N; j++) {
        const g = dy[yo + j];
        acc += g * W[wo + j];
        dW[wo + j] += xv * g;
      }
      if (dx) dx[xo + k] += acc;
    }
    if (db) for (let j = 0; j < N; j++) db[j] += dy[yo + j];
  }
}

function layernormFwd(
  x: Float32Array, M: number, C: number, g: Float32Array, b: Float32Array,
  y: Float32Array, xhat: Float32Array, rstd: Float32Array, eps: number
): void {
  for (let i = 0; i < M; i++) {
    const off = i * C;
    let m = 0;
    for (let j = 0; j < C; j++) m += x[off + j];
    m /= C;
    let v = 0;
    for (let j = 0; j < C; j++) {
      const d = x[off + j] - m;
      v += d * d;
    }
    v /= C;
    const r = 1 / Math.sqrt(v + eps);
    rstd[i] = r;
    for (let j = 0; j < C; j++) {
      const xh = (x[off + j] - m) * r;
      xhat[off + j] = xh;
      y[off + j] = xh * g[j] + b[j];
    }
  }
}

/** Accumulates dg/db and adds into dx. */
function layernormBwd(
  dy: Float32Array, xhat: Float32Array, rstd: Float32Array, M: number, C: number,
  g: Float32Array, dg: Float32Array | null, db: Float32Array | null, dx: Float32Array | null
): void {
  for (let i = 0; i < M; i++) {
    const off = i * C;
    const r = rstd[i];
    let s1 = 0;
    let s2 = 0;
    for (let j = 0; j < C; j++) {
      const t = dy[off + j] * g[j];
      s1 += t;
      s2 += t * xhat[off + j];
    }
    const c1 = s1 / C;
    const c2 = s2 / C;
    for (let j = 0; j < C; j++) {
      const t = dy[off + j] * g[j];
      if (dx) dx[off + j] += r * (t - c1 - xhat[off + j] * c2);
      if (dg) dg[j] += dy[off + j] * xhat[off + j];
      if (db) db[j] += dy[off + j];
    }
  }
}

const GELU_C = Math.sqrt(2 / Math.PI);
const GELU_A = 0.044715;

function gelu(x: number): number {
  const t = Math.tanh(GELU_C * (x + GELU_A * x * x * x));
  return 0.5 * x * (1 + t);
}

/** dx += dy · gelu'(x) */
function geluBwd(dy: Float32Array, x: Float32Array, dx: Float32Array, n: number): void {
  for (let i = 0; i < n; i++) {
    const xv = x[i];
    const t = Math.tanh(GELU_C * (xv + GELU_A * xv * xv * xv));
    const du = GELU_C * (1 + 3 * GELU_A * xv * xv);
    dx[i] += dy[i] * (0.5 * (1 + t) + 0.5 * xv * (1 - t * t) * du);
  }
}

/** Causal multi-head self-attention. qkv is [M,3C]; probs is [H,M,M]. */
function attnFwd(
  qkv: Float32Array, probs: Float32Array, M: number, C: number, H: number, y: Float32Array
): void {
  const hd = C / H;
  const scale = 1 / Math.sqrt(hd);
  const row = 3 * C;
  for (let h = 0; h < H; h++) {
    const qo = h * hd;
    const ko = C + h * hd;
    const vo = 2 * C + h * hd;
    const pb = h * M * M;
    for (let i = 0; i < M; i++) {
      const qBase = i * row + qo;
      let max = -Infinity;
      for (let j = 0; j <= i; j++) {
        const kBase = j * row + ko;
        let s = 0;
        for (let d = 0; d < hd; d++) s += qkv[qBase + d] * qkv[kBase + d];
        s *= scale;
        probs[pb + i * M + j] = s;
        if (s > max) max = s;
      }
      let sum = 0;
      for (let j = 0; j <= i; j++) {
        const e = Math.exp(probs[pb + i * M + j] - max);
        probs[pb + i * M + j] = e;
        sum += e;
      }
      const inv = 1 / sum;
      for (let j = 0; j <= i; j++) probs[pb + i * M + j] *= inv;
      for (let j = i + 1; j < M; j++) probs[pb + i * M + j] = 0;
      const yo = i * C + qo;
      for (let d = 0; d < hd; d++) y[yo + d] = 0;
      for (let j = 0; j <= i; j++) {
        const p = probs[pb + i * M + j];
        if (p === 0) continue;
        const vBase = j * row + vo;
        for (let d = 0; d < hd; d++) y[yo + d] += p * qkv[vBase + d];
      }
    }
  }
}

/** Accumulates into dqkv. `dRow` is scratch of length M. */
function attnBwd(
  dy: Float32Array, qkv: Float32Array, probs: Float32Array, M: number, C: number,
  H: number, dqkv: Float32Array, dRow: Float32Array
): void {
  const hd = C / H;
  const scale = 1 / Math.sqrt(hd);
  const row = 3 * C;
  for (let h = 0; h < H; h++) {
    const qo = h * hd;
    const ko = C + h * hd;
    const vo = 2 * C + h * hd;
    const pb = h * M * M;
    for (let i = 0; i < M; i++) {
      const yo = i * C + qo;
      const qBase = i * row + qo;
      // dP row, plus P's contribution to dV
      for (let j = 0; j <= i; j++) {
        const vBase = j * row + vo;
        let acc = 0;
        for (let d = 0; d < hd; d++) acc += dy[yo + d] * qkv[vBase + d];
        dRow[j] = acc;
        const p = probs[pb + i * M + j];
        if (p !== 0) for (let d = 0; d < hd; d++) dqkv[vBase + d] += p * dy[yo + d];
      }
      // softmax backward for this row: dS = P ⊙ (dP − Σ dP·P)
      let dot = 0;
      for (let j = 0; j <= i; j++) dot += dRow[j] * probs[pb + i * M + j];
      for (let j = 0; j <= i; j++) {
        const p = probs[pb + i * M + j];
        const ds = p * (dRow[j] - dot) * scale;
        if (ds === 0) continue;
        const kBase = j * row + ko;
        for (let d = 0; d < hd; d++) {
          dqkv[qBase + d] += ds * qkv[kBase + d];
          dqkv[kBase + d] += ds * qkv[qBase + d];
        }
      }
    }
  }
}

/* ------------------------------------------------------------------ *
 * The trainer
 * ------------------------------------------------------------------ */

class Param {
  readonly name: string;
  readonly w: Float32Array;
  readonly g: Float32Array;
  readonly m: Float32Array;
  readonly v: Float32Array;
  readonly decay: number;

  constructor(name: string, size: number, decay = 0) {
    this.name = name;
    this.w = new Float32Array(size);
    this.g = new Float32Array(size);
    this.m = new Float32Array(size);
    this.v = new Float32Array(size);
    this.decay = decay;
  }

  zero(): void {
    this.g.fill(0);
  }
}

interface Layer {
  ln1g: Param; ln1b: Param;
  qkv: Param; qkvb: Param;
  proj: Param; projb: Param;
  ln2g: Param; ln2b: Param;
  fc1: Param; fc1b: Param;
  fc2: Param; fc2b: Param;
}

/** Per-layer activations. Every layer needs its own copy — see `forward`. */
interface Activations {
  xln1: Float32Array; xhat1: Float32Array; r1: Float32Array;
  qkv: Float32Array; probs: Float32Array; attnY: Float32Array;
  xln2: Float32Array; xhat2: Float32Array; r2: Float32Array;
  fc1: Float32Array; fc1g: Float32Array;
}

/** Scratch reused inside one forward/backward pair. */
interface Buffers {
  x: Float32Array; attnP: Float32Array; h1: Float32Array; fc2: Float32Array;
  xf: Float32Array; xhatf: Float32Array; rf: Float32Array;
  logits: Float32Array; dRow: Float32Array; dlogits: Float32Array;
  da: Float32Array; dA: Float32Array; dB: Float32Array; dC: Float32Array;
  dG: Float32Array; dFC1: Float32Array; dqkv: Float32Array;
}

export interface TrainerOptions {
  data: Int32Array;
  vocab: Vocab;
  config: ModelConfig;
  seed?: number;
}

export class Trainer {
  readonly data: Int32Array;
  readonly vocab: Vocab;
  readonly config: ModelConfig;
  readonly seed: number;
  rand: () => number;

  readonly C: number;
  readonly H: number;
  readonly F: number;
  readonly T: number;
  readonly B: number;
  readonly eps: number;
  readonly vocabSize: number;

  readonly wte: Param;
  readonly wpe: Param;
  readonly layers: Layer[];
  readonly lnfg: Param;
  readonly lnfb: Param;
  readonly head: Param;
  readonly headb: Param;
  readonly params: Param[];

  readonly acts: Activations[];
  readonly w: Buffers;

  stepCount = 0;
  tokensSeen = 0;
  lossHistory: number[] = [];
  smoothedLoss: number | null = null;
  lastWindow = 0;
  lastM = 0;
  gradDiv: number;

  constructor({ data, vocab, config, seed = 1337 }: TrainerOptions) {
    if (!data || data.length < 2) throw new Error('The corpus is too short to train on.');
    this.data = data;
    this.vocab = vocab;
    this.config = { ...config };
    this.seed = seed >>> 0;
    this.rand = mulberry32(this.seed);

    const C = (this.C = this.config.dModel);
    const H = (this.H = this.config.nHead);
    const F = (this.F = this.config.dFF);
    const B = (this.B = this.config.batchSize);
    let T = this.config.blockSize;
    if (data.length <= T) T = Math.max(2, Math.min(T, data.length - 1));
    this.T = T;
    this.config.blockSize = T;
    this.eps = 1e-5;
    const Vn = (this.vocabSize = vocab.size);
    const nLayer = this.config.nLayer;
    const wd = this.config.weightDecay || 0;

    this.wte = new Param('wte', Vn * C);
    this.wpe = new Param('wpe', T * C);
    this.layers = [];
    for (let l = 0; l < nLayer; l++) {
      this.layers.push({
        ln1g: new Param(`h${l}.ln1.g`, C),
        ln1b: new Param(`h${l}.ln1.b`, C),
        qkv: new Param(`h${l}.qkv.w`, C * 3 * C, wd),
        qkvb: new Param(`h${l}.qkv.b`, 3 * C),
        proj: new Param(`h${l}.proj.w`, C * C, wd),
        projb: new Param(`h${l}.proj.b`, C),
        ln2g: new Param(`h${l}.ln2.g`, C),
        ln2b: new Param(`h${l}.ln2.b`, C),
        fc1: new Param(`h${l}.fc1.w`, C * F, wd),
        fc1b: new Param(`h${l}.fc1.b`, F),
        fc2: new Param(`h${l}.fc2.w`, F * C, wd),
        fc2b: new Param(`h${l}.fc2.b`, C),
      });
    }
    this.lnfg = new Param('lnf.g', C);
    this.lnfb = new Param('lnf.b', C);
    this.head = new Param('head.w', C * Vn, wd);
    this.headb = new Param('head.b', Vn);

    this.params = [this.wte, this.wpe];
    for (const L of this.layers) for (const p of Object.values(L)) this.params.push(p);
    this.params.push(this.lnfg, this.lnfb, this.head, this.headb);

    this.#initWeights();

    const a = (n: number): Float32Array => new Float32Array(n);

    // Per-layer activations. Every layer needs its own copy: a single shared set
    // of buffers would be overwritten by each layer in the forward pass, and the
    // backward pass for layer 0 would then read layer 1's numbers.
    this.acts = [];
    for (let l = 0; l < nLayer; l++) {
      this.acts.push({
        xln1: a(T * C), xhat1: a(T * C), r1: a(T),
        qkv: a(T * 3 * C), probs: a(H * T * T), attnY: a(T * C),
        xln2: a(T * C), xhat2: a(T * C), r2: a(T),
        fc1: a(T * F), fc1g: a(T * F),
      });
    }

    this.w = {
      x: a(T * C), attnP: a(T * C), h1: a(T * C), fc2: a(T * C),
      xf: a(T * C), xhatf: a(T * C), rf: a(T),
      logits: a(T * Vn), dRow: a(T), dlogits: a(T * Vn),
      da: a(T * C), dA: a(T * C), dB: a(T * C), dC: a(T * C),
      dG: a(T * F), dFC1: a(T * F), dqkv: a(T * 3 * C),
    };

    this.gradDiv = B * T;
  }

  #initWeights(): void {
    const scale = 1 / Math.sqrt(2 * this.config.nLayer);
    const initMatrix = (p: Param, std: number): void => {
      for (let i = 0; i < p.w.length; i++) p.w[i] = gaussian(this.rand) * std;
    };
    initMatrix(this.wte, 0.02);
    initMatrix(this.wpe, 0.02);
    for (const L of this.layers) {
      fill(L.ln1g.w, 1); fill(L.ln1b.w, 0);
      fill(L.ln2g.w, 1); fill(L.ln2b.w, 0);
      initMatrix(L.qkv, 0.02); fill(L.qkvb.w, 0);
      initMatrix(L.proj, 0.02 * scale); fill(L.projb.w, 0);
      initMatrix(L.fc1, 0.02); fill(L.fc1b.w, 0);
      initMatrix(L.fc2, 0.02 * scale); fill(L.fc2b.w, 0);
    }
    fill(this.lnfg.w, 1); fill(this.lnfb.w, 0);
    initMatrix(this.head, 0.02); fill(this.headb.w, 0);
  }

  get paramCount(): number {
    let n = 0;
    for (const p of this.params) n += p.w.length;
    return n;
  }

  /* ---------------- forward ---------------- */

  /** Forward over ids[offset .. offset+M). Returns mean cross-entropy (0 if !wantLoss). */
  forward(offset: number, M: number, ids: Int32Array = this.data, wantLoss = true): number {
    const C = this.C, H = this.H, F = this.F;
    const Vn = this.vocabSize;
    const w = this.w;
    const wte = this.wte.w, wpe = this.wpe.w;

    for (let t = 0; t < M; t++) {
      const tok = ids[offset + t];
      const eo = tok * C, xo = t * C;
      for (let j = 0; j < C; j++) w.x[xo + j] = wte[eo + j] + wpe[t * C + j];
    }

    for (let l = 0; l < this.layers.length; l++) {
      const L = this.layers[l];
      const A = this.acts[l];
      const nMC = M * C;
      const nMF = M * F;
      layernormFwd(w.x, M, C, L.ln1g.w, L.ln1b.w, A.xln1, A.xhat1, A.r1, this.eps);
      linearFwd(A.xln1, M, C, L.qkv.w, 3 * C, L.qkvb.w, A.qkv);
      attnFwd(A.qkv, A.probs, M, C, H, A.attnY);
      linearFwd(A.attnY, M, C, L.proj.w, C, L.projb.w, w.attnP);
      for (let i = 0; i < nMC; i++) w.h1[i] = w.x[i] + w.attnP[i];

      layernormFwd(w.h1, M, C, L.ln2g.w, L.ln2b.w, A.xln2, A.xhat2, A.r2, this.eps);
      linearFwd(A.xln2, M, C, L.fc1.w, F, L.fc1b.w, A.fc1);
      for (let i = 0; i < nMF; i++) A.fc1g[i] = gelu(A.fc1[i]);
      linearFwd(A.fc1g, M, F, L.fc2.w, C, L.fc2b.w, w.fc2);
      for (let i = 0; i < nMC; i++) w.x[i] = w.h1[i] + w.fc2[i];
    }

    layernormFwd(w.x, M, C, this.lnfg.w, this.lnfb.w, w.xf, w.xhatf, w.rf, this.eps);
    linearFwd(w.xf, M, C, this.head.w, Vn, this.headb.w, w.logits);

    if (!wantLoss || M === 0) return 0;

    let loss = 0;
    const d = w.dlogits;
    for (let t = 0; t < M; t++) {
      const lo = t * Vn;
      let max = -Infinity;
      for (let j = 0; j < Vn; j++) if (w.logits[lo + j] > max) max = w.logits[lo + j];
      let sum = 0;
      for (let j = 0; j < Vn; j++) {
        const e = Math.exp(w.logits[lo + j] - max);
        d[lo + j] = e;
        sum += e;
      }
      const inv = 1 / sum;
      const target = ids[offset + t + 1];
      loss += -Math.log(Math.max(d[lo + target] * inv, 1e-30));
      for (let j = 0; j < Vn; j++) d[lo + j] *= inv;
      d[lo + target] -= 1;
    }
    const invM = 1 / M;
    const n = M * Vn;
    for (let i = 0; i < n; i++) d[i] *= invM;
    return loss * invM;
  }

  /** Backward for the window the last forward() saw (uses w.dlogits). */
  backward(offset: number, M: number, ids: Int32Array = this.data): void {
    const C = this.C, H = this.H, F = this.F;
    const Vn = this.vocabSize;
    const w = this.w;
    const nMC = M * C;
    const nMF = M * F;

    fill(w.dB, 0, nMC);
    linearBwd(w.dlogits, w.xf, M, C, this.head.w, Vn, this.head.g, this.headb.g, w.dB);
    fill(w.da, 0, nMC);
    layernormBwd(w.dB, w.xhatf, w.rf, M, C, this.lnfg.w, this.lnfg.g, this.lnfb.g, w.da);

    for (let l = this.layers.length - 1; l >= 0; l--) {
      const L = this.layers[l];
      const A = this.acts[l];

      // feed-forward + its residual
      fill(w.dG, 0, nMF);
      linearBwd(w.da, A.fc1g, M, F, L.fc2.w, C, L.fc2.g, L.fc2b.g, w.dG);
      fill(w.dFC1, 0, nMF);
      geluBwd(w.dG, A.fc1, w.dFC1, nMF);
      fill(w.dB, 0, nMC);
      linearBwd(w.dFC1, A.xln2, M, C, L.fc1.w, F, L.fc1.g, L.fc1b.g, w.dB);
      fill(w.dA, 0, nMC);
      layernormBwd(w.dB, A.xhat2, A.r2, M, C, L.ln2g.w, L.ln2g.g, L.ln2b.g, w.dA);
      addInto(w.dA, w.da, nMC);

      // attention + its residual
      fill(w.dB, 0, nMC);
      linearBwd(w.dA, A.attnY, M, C, L.proj.w, C, L.proj.g, L.projb.g, w.dB);
      fill(w.dqkv, 0, M * 3 * C);
      attnBwd(w.dB, A.qkv, A.probs, M, C, H, w.dqkv, w.dRow);
      fill(w.dC, 0, nMC);
      linearBwd(w.dqkv, A.xln1, M, C, L.qkv.w, 3 * C, L.qkv.g, L.qkvb.g, w.dC);
      fill(w.da, 0, nMC);
      layernormBwd(w.dC, A.xhat1, A.r1, M, C, L.ln1g.w, L.ln1g.g, L.ln1b.g, w.da);
      addInto(w.da, w.dA, nMC);
    }

    for (let t = 0; t < M; t++) {
      const tok = ids[offset + t];
      const eo = tok * C, xo = t * C;
      for (let j = 0; j < C; j++) {
        const g = w.da[xo + j];
        this.wte.g[eo + j] += g;
        this.wpe.g[t * C + j] += g;
      }
    }
  }

  /* ---------------- training ---------------- */

  zeroGrads(): void {
    for (const p of this.params) p.zero();
  }

  /** One forward+backward over a window, accumulating (unnormalised) gradients. */
  accumulate(offset: number): number {
    const M = Math.min(this.T, this.data.length - 1 - offset);
    this.lastWindow = 0;
    if (M <= 0) return 0;
    const loss = this.forward(offset, M);
    this.backward(offset, M);
    this.lastWindow = M;
    return loss;
  }

  /** One AdamW step over a fresh random batch. */
  step(): StepResult {
    const T = this.T, B = this.B;
    const max = this.data.length - 1 - T;
    this.zeroGrads();
    let total = 0, count = 0, tokens = 0;
    for (let b = 0; b < B; b++) {
      const offset = max > 0 ? Math.floor(this.rand() * (max + 1)) : 0;
      const loss = this.accumulate(offset);
      if (this.lastWindow > 0) {
        total += loss;
        count++;
        tokens += this.lastWindow;
      }
    }
    this.gradDiv = Math.max(1, tokens);
    this.adamW();
    this.stepCount++;
    this.tokensSeen += tokens;
    const mean = count > 0 ? total / count : 0;
    this.smoothedLoss = this.smoothedLoss === null ? mean : this.smoothedLoss * 0.9 + mean * 0.1;
    this.lossHistory.push(mean);
    return { loss: mean, tokens };
  }

  adamW(): void {
    const t = this.stepCount + 1;
    const b1 = 0.9, b2 = 0.999, eps = 1e-8;
    const bc1 = 1 - Math.pow(b1, t);
    const bc2 = 1 - Math.pow(b2, t);
    const lr = this.config.lr;
    const div = this.gradDiv;
    for (const p of this.params) {
      const wv = p.w, g = p.g, m = p.m, v = p.v;
      const decay = p.decay ? 1 - lr * p.decay : 1;
      for (let i = 0; i < wv.length; i++) {
        const gi = g[i] / div;
        const mi = (m[i] = b1 * m[i] + (1 - b1) * gi);
        const vi = (v[i] = b2 * v[i] + (1 - b2) * gi * gi);
        const upd = mi / bc1 / (Math.sqrt(vi / bc2) + eps);
        wv[i] = wv[i] * decay - lr * upd;
      }
    }
  }

  /** Loss over a fixed window — used by the gradient check. */
  lossAt(offset: number): number {
    const M = Math.min(this.T, this.data.length - 1 - offset);
    return this.forward(offset, M);
  }

  /** Fill gradients for a fixed window with the given divisor (gradient check). */
  gradsAt(offset: number, div = 1): number {
    this.gradDiv = div;
    this.zeroGrads();
    const M = Math.min(this.T, this.data.length - 1 - offset);
    this.forward(offset, M);
    this.backward(offset, M);
    return M;
  }

  /* ---------------- generation ---------------- */

  #pick(temperature: number, topK: number): number {
    const Vn = this.vocabSize;
    const base = (this.lastM - 1) * Vn;
    const temp = Math.max(temperature, 1e-4);
    const scores = new Float64Array(Vn);
    for (let j = 0; j < Vn; j++) scores[j] = this.w.logits[base + j] / temp;

    let pool: number[];
    let max = -Infinity;
    if (topK > 0 && topK < Vn) {
      pool = Array.from({ length: Vn }, (_, j) => j)
        .sort((a, b) => scores[b] - scores[a])
        .slice(0, topK);
      for (const j of pool) if (scores[j] > max) max = scores[j];
    } else {
      pool = Array.from({ length: Vn }, (_, j) => j);
      for (let j = 0; j < Vn; j++) if (scores[j] > max) max = scores[j];
    }

    let sum = 0;
    const probs = new Float64Array(pool.length);
    for (let i = 0; i < pool.length; i++) {
      const e = Math.exp(scores[pool[i]] - max);
      probs[i] = e;
      sum += e;
    }
    let r = this.rand() * sum;
    for (let i = 0; i < pool.length; i++) {
      r -= probs[i];
      if (r <= 0) return pool[i];
    }
    return pool[pool.length - 1];
  }

  /**
   * Generate text. Returns the characters produced (the prompt is not included).
   *
   * `topK` defaults to truncation, not to 0 (off). A top-k of zero draws from the
   * entire vocabulary, so the long tail of near-impossible characters is sampled
   * constantly and the text is noisier for no gain — the page's control is for
   * WIDENING it, not for switching it off. 20 is a sensible cut for a character
   * vocabulary; the loss and word measurements in `PRESETS` were taken at 40, the
   * looser setting, so the default here is the more conservative of the two.
   */
  sample({ prompt = '', length = 240, temperature = 0.8, topK = 20, seed = null }: SampleOptions = {}): string {
    if (seed !== null) this.rand = mulberry32(seed);
    const T = this.T;
    const ids: number[] = prompt
      ? Array.from(encode(prompt, this.vocab))
      : [lineStartToken(this.data, this.vocab)];
    const out = ids.slice();
    const maxNewlines = Math.max(1, Math.round(length / 24));
    let newlines = 0;
    for (let i = 0; i < length; i++) {
      const ctx = out.slice(Math.max(0, out.length - T));
      const M = ctx.length;
      this.forward(0, M, Int32Array.from(ctx), false);
      this.lastM = M;
      const next = this.#pick(temperature, topK);
      out.push(next);
      if (this.vocab.chars[next] === '\n' && ++newlines >= maxNewlines) break;
    }
    return decode(out.slice(ids.length), this.vocab);
  }

  /* ---------------- export ---------------- */

  exportWeights(): ExportedWeights {
    const params: Record<string, number[]> = {};
    for (const p of this.params) params[p.name] = Array.from(p.w, (x: number) => Number(x.toFixed(6)));
    return {
      name: 'llm-demo',
      architecture:
        'character-level GPT: token + position embeddings, causal multi-head self-attention, feed-forward block, layer norm, residual connections',
      config: { ...this.config },
      vocab: this.vocab.chars,
      params,
    };
  }

  exportReport(extra: Record<string, unknown> = {}): TrainingReport {
    const history = this.lossHistory;
    const stride = Math.max(1, Math.floor(history.length / 60));
    const curve = history.filter((_, i) => i % stride === 0 || i === history.length - 1);
    return {
      site: 'https://llm-demo.nodejavascript.com',
      generated_at: new Date().toISOString(),
      model: {
        parameters: this.paramCount,
        layers: this.layers.length,
        heads: this.H,
        width: this.C,
        feed_forward: this.F,
        context: this.T,
        batch_size: this.B,
        learning_rate: this.config.lr,
        optimizer: 'AdamW',
        weight_decay: this.config.weightDecay,
        tokenizer: 'character level',
        vocabulary_size: this.vocabSize,
      },
      training: {
        steps: this.stepCount,
        tokens_seen: this.tokensSeen,
        corpus_characters: this.data.length,
        final_loss: history.length ? history[history.length - 1] : null,
        smoothed_loss: this.smoothedLoss,
        curve,
      },
      ...extra,
    };
  }
}
