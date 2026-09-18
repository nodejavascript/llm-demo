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
 * Random numbers (seeded, so runs are reproducible and testable)
 * ------------------------------------------------------------------ */
export function mulberry32(seed) {
    let a = seed >>> 0;
    return function next() {
        a = (a + 0x6d2b79f5) | 0;
        let t = Math.imul(a ^ (a >>> 15), 1 | a);
        t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
        return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
    };
}
function gaussian(rand) {
    let u = 0;
    while (u === 0)
        u = rand();
    return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * rand());
}
/* ------------------------------------------------------------------ *
 * Tokenizer — character level, built from the corpus itself
 * ------------------------------------------------------------------ */
function codePointCount(text) {
    let n = 0;
    for (let i = 0; i < text.length; i++) {
        const c = text.charCodeAt(i);
        if (c >= 0xd800 && c <= 0xdbff && i + 1 < text.length)
            i++;
        n++;
    }
    return n;
}
function pickUnusedChar(text, taken) {
    for (let cp = 0xe000; cp <= 0xf8ff; cp++) {
        const ch = String.fromCodePoint(cp);
        if (!taken.has(ch) && !text.includes(ch))
            return ch;
    }
    return '\u0000';
}
/**
 * Build the character vocabulary from the corpus. The most frequent characters
 * are kept; anything beyond `maxVocab` is folded into one "unknown" bucket, so
 * a corpus full of emoji cannot blow the model up.
 */
export function buildVocab(text, maxVocab = 128) {
    const freq = new Map();
    for (const ch of text)
        freq.set(ch, (freq.get(ch) ?? 0) + 1);
    let chars = [...freq.keys()].sort((a, b) => (freq.get(b) ?? 0) - (freq.get(a) ?? 0) || (a < b ? -1 : a > b ? 1 : 0));
    let unknown = null;
    if (chars.length > maxVocab) {
        const keep = new Set(chars.slice(0, maxVocab - 1));
        unknown = pickUnusedChar(text, keep);
        chars = chars.slice(0, maxVocab - 1);
        chars.push(unknown);
    }
    const stoi = new Map();
    chars.forEach((ch, i) => stoi.set(ch, i));
    return { chars, stoi, size: chars.length, unkIndex: unknown ? chars.length - 1 : -1 };
}
export function encode(text, vocab) {
    const out = new Int32Array(codePointCount(text));
    let i = 0;
    for (const ch of text) {
        const id = vocab.stoi.get(ch);
        out[i++] = id === undefined ? vocab.unkIndex : id;
    }
    return out;
}
export function decode(ids, vocab) {
    const out = [];
    for (let i = 0; i < ids.length; i++)
        out.push(vocab.chars[ids[i]] ?? '');
    return out.join('');
}
/** The character a line most often starts with — the seed for an empty prompt. */
export function lineStartToken(data, vocab) {
    const counts = new Int32Array(vocab.size);
    let prev = -1;
    for (let i = 0; i < data.length; i++) {
        if (prev === -1 || vocab.chars[prev] === '\n')
            counts[data[i]]++;
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
 * `seconds` is the time the FIRST run in a fresh tab takes — the one a visitor
 * actually meets — because the browser has to compile the loops before it
 * settles. Measured in Chrome on a desktop CPU, 2026-09-17: 600 `quick` steps in
 * 13.8 s cold and 8 s warm, 250 `standard` steps in 27 s warm, and the biggest
 * model at roughly 0.4 s a step. The page also reports the rate it is really
 * achieving, with an estimate taken from that rate, so a slower machine says so
 * rather than quietly taking longer. Any run can be stopped early and tested as
 * it stands.
 */
export const PRESETS = {
    quick: {
        key: 'quick', label: 'Quick', blurb: 'seconds — enough to watch it learn',
        seconds: 15,
        nLayer: 1, nHead: 2, dModel: 32, dFF: 64,
        blockSize: 24, batchSize: 12, lr: 3e-3, steps: 600, weightDecay: 0.01,
    },
    standard: {
        key: 'standard', label: 'Standard', blurb: 'about half a minute — clearly better',
        seconds: 45,
        nLayer: 2, nHead: 2, dModel: 64, dFF: 128,
        blockSize: 32, batchSize: 12, lr: 2e-3, steps: 250, weightDecay: 0.01,
    },
    thorough: {
        key: 'thorough', label: 'Thorough', blurb: 'a couple of minutes — the biggest that stays usable here',
        seconds: 150,
        nLayer: 3, nHead: 4, dModel: 96, dFF: 192,
        blockSize: 48, batchSize: 12, lr: 1.5e-3, steps: 300, weightDecay: 0.01,
    },
};
export function preset(name) {
    const found = PRESETS[name];
    return { ...(found ?? PRESETS.quick) };
}
export function parameterCount(config, vocabSize) {
    const { nLayer, dModel: C, dFF: F, blockSize: T } = config;
    const perLayer = 3 * C * C + 3 * C + C * C + C + C * F + F + F * C + C + 4 * C;
    return vocabSize * C + T * C + nLayer * perLayer + 2 * C + C * vocabSize + vocabSize;
}
/* ------------------------------------------------------------------ *
 * Kernels
 * ------------------------------------------------------------------ */
function fill(arr, value, n = arr.length) {
    for (let i = 0; i < n; i++)
        arr[i] = value;
}
function addInto(dst, src, n) {
    for (let i = 0; i < n; i++)
        dst[i] += src[i];
}
/** y[M,N] = x[M,K] · W[K,N] + b[N] */
function linearFwd(x, M, K, W, N, b, y) {
    for (let i = 0; i < M; i++) {
        const yo = i * N;
        const xo = i * K;
        if (b)
            for (let j = 0; j < N; j++)
                y[yo + j] = b[j];
        else
            for (let j = 0; j < N; j++)
                y[yo + j] = 0;
        for (let k = 0; k < K; k++) {
            const xv = x[xo + k];
            if (xv === 0)
                continue;
            const wo = k * N;
            for (let j = 0; j < N; j++)
                y[yo + j] += xv * W[wo + j];
        }
    }
}
/** Accumulates dW += xᵀ·dy, db += Σᵢ dy, and optionally dx += dy·Wᵀ. */
function linearBwd(dy, x, M, K, W, N, dW, db, dx) {
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
            if (dx)
                dx[xo + k] += acc;
        }
        if (db)
            for (let j = 0; j < N; j++)
                db[j] += dy[yo + j];
    }
}
function layernormFwd(x, M, C, g, b, y, xhat, rstd, eps) {
    for (let i = 0; i < M; i++) {
        const off = i * C;
        let m = 0;
        for (let j = 0; j < C; j++)
            m += x[off + j];
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
function layernormBwd(dy, xhat, rstd, M, C, g, dg, db, dx) {
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
            if (dx)
                dx[off + j] += r * (t - c1 - xhat[off + j] * c2);
            if (dg)
                dg[j] += dy[off + j] * xhat[off + j];
            if (db)
                db[j] += dy[off + j];
        }
    }
}
const GELU_C = Math.sqrt(2 / Math.PI);
const GELU_A = 0.044715;
function gelu(x) {
    const t = Math.tanh(GELU_C * (x + GELU_A * x * x * x));
    return 0.5 * x * (1 + t);
}
/** dx += dy · gelu'(x) */
function geluBwd(dy, x, dx, n) {
    for (let i = 0; i < n; i++) {
        const xv = x[i];
        const t = Math.tanh(GELU_C * (xv + GELU_A * xv * xv * xv));
        const du = GELU_C * (1 + 3 * GELU_A * xv * xv);
        dx[i] += dy[i] * (0.5 * (1 + t) + 0.5 * xv * (1 - t * t) * du);
    }
}
/** Causal multi-head self-attention. qkv is [M,3C]; probs is [H,M,M]. */
function attnFwd(qkv, probs, M, C, H, y) {
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
                for (let d = 0; d < hd; d++)
                    s += qkv[qBase + d] * qkv[kBase + d];
                s *= scale;
                probs[pb + i * M + j] = s;
                if (s > max)
                    max = s;
            }
            let sum = 0;
            for (let j = 0; j <= i; j++) {
                const e = Math.exp(probs[pb + i * M + j] - max);
                probs[pb + i * M + j] = e;
                sum += e;
            }
            const inv = 1 / sum;
            for (let j = 0; j <= i; j++)
                probs[pb + i * M + j] *= inv;
            for (let j = i + 1; j < M; j++)
                probs[pb + i * M + j] = 0;
            const yo = i * C + qo;
            for (let d = 0; d < hd; d++)
                y[yo + d] = 0;
            for (let j = 0; j <= i; j++) {
                const p = probs[pb + i * M + j];
                if (p === 0)
                    continue;
                const vBase = j * row + vo;
                for (let d = 0; d < hd; d++)
                    y[yo + d] += p * qkv[vBase + d];
            }
        }
    }
}
/** Accumulates into dqkv. `dRow` is scratch of length M. */
function attnBwd(dy, qkv, probs, M, C, H, dqkv, dRow) {
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
                for (let d = 0; d < hd; d++)
                    acc += dy[yo + d] * qkv[vBase + d];
                dRow[j] = acc;
                const p = probs[pb + i * M + j];
                if (p !== 0)
                    for (let d = 0; d < hd; d++)
                        dqkv[vBase + d] += p * dy[yo + d];
            }
            // softmax backward for this row: dS = P ⊙ (dP − Σ dP·P)
            let dot = 0;
            for (let j = 0; j <= i; j++)
                dot += dRow[j] * probs[pb + i * M + j];
            for (let j = 0; j <= i; j++) {
                const p = probs[pb + i * M + j];
                const ds = p * (dRow[j] - dot) * scale;
                if (ds === 0)
                    continue;
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
    name;
    w;
    g;
    m;
    v;
    decay;
    constructor(name, size, decay = 0) {
        this.name = name;
        this.w = new Float32Array(size);
        this.g = new Float32Array(size);
        this.m = new Float32Array(size);
        this.v = new Float32Array(size);
        this.decay = decay;
    }
    zero() {
        this.g.fill(0);
    }
}
export class Trainer {
    data;
    vocab;
    config;
    seed;
    rand;
    C;
    H;
    F;
    T;
    B;
    eps;
    vocabSize;
    wte;
    wpe;
    layers;
    lnfg;
    lnfb;
    head;
    headb;
    params;
    acts;
    w;
    stepCount = 0;
    tokensSeen = 0;
    lossHistory = [];
    smoothedLoss = null;
    lastWindow = 0;
    lastM = 0;
    gradDiv;
    constructor({ data, vocab, config, seed = 1337 }) {
        if (!data || data.length < 2)
            throw new Error('The corpus is too short to train on.');
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
        if (data.length <= T)
            T = Math.max(2, Math.min(T, data.length - 1));
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
        for (const L of this.layers)
            for (const p of Object.values(L))
                this.params.push(p);
        this.params.push(this.lnfg, this.lnfb, this.head, this.headb);
        this.#initWeights();
        const a = (n) => new Float32Array(n);
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
    #initWeights() {
        const scale = 1 / Math.sqrt(2 * this.config.nLayer);
        const initMatrix = (p, std) => {
            for (let i = 0; i < p.w.length; i++)
                p.w[i] = gaussian(this.rand) * std;
        };
        initMatrix(this.wte, 0.02);
        initMatrix(this.wpe, 0.02);
        for (const L of this.layers) {
            fill(L.ln1g.w, 1);
            fill(L.ln1b.w, 0);
            fill(L.ln2g.w, 1);
            fill(L.ln2b.w, 0);
            initMatrix(L.qkv, 0.02);
            fill(L.qkvb.w, 0);
            initMatrix(L.proj, 0.02 * scale);
            fill(L.projb.w, 0);
            initMatrix(L.fc1, 0.02);
            fill(L.fc1b.w, 0);
            initMatrix(L.fc2, 0.02 * scale);
            fill(L.fc2b.w, 0);
        }
        fill(this.lnfg.w, 1);
        fill(this.lnfb.w, 0);
        initMatrix(this.head, 0.02);
        fill(this.headb.w, 0);
    }
    get paramCount() {
        let n = 0;
        for (const p of this.params)
            n += p.w.length;
        return n;
    }
    /* ---------------- forward ---------------- */
    /** Forward over ids[offset .. offset+M). Returns mean cross-entropy (0 if !wantLoss). */
    forward(offset, M, ids = this.data, wantLoss = true) {
        const C = this.C, H = this.H, F = this.F;
        const Vn = this.vocabSize;
        const w = this.w;
        const wte = this.wte.w, wpe = this.wpe.w;
        for (let t = 0; t < M; t++) {
            const tok = ids[offset + t];
            const eo = tok * C, xo = t * C;
            for (let j = 0; j < C; j++)
                w.x[xo + j] = wte[eo + j] + wpe[t * C + j];
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
            for (let i = 0; i < nMC; i++)
                w.h1[i] = w.x[i] + w.attnP[i];
            layernormFwd(w.h1, M, C, L.ln2g.w, L.ln2b.w, A.xln2, A.xhat2, A.r2, this.eps);
            linearFwd(A.xln2, M, C, L.fc1.w, F, L.fc1b.w, A.fc1);
            for (let i = 0; i < nMF; i++)
                A.fc1g[i] = gelu(A.fc1[i]);
            linearFwd(A.fc1g, M, F, L.fc2.w, C, L.fc2b.w, w.fc2);
            for (let i = 0; i < nMC; i++)
                w.x[i] = w.h1[i] + w.fc2[i];
        }
        layernormFwd(w.x, M, C, this.lnfg.w, this.lnfb.w, w.xf, w.xhatf, w.rf, this.eps);
        linearFwd(w.xf, M, C, this.head.w, Vn, this.headb.w, w.logits);
        if (!wantLoss || M === 0)
            return 0;
        let loss = 0;
        const d = w.dlogits;
        for (let t = 0; t < M; t++) {
            const lo = t * Vn;
            let max = -Infinity;
            for (let j = 0; j < Vn; j++)
                if (w.logits[lo + j] > max)
                    max = w.logits[lo + j];
            let sum = 0;
            for (let j = 0; j < Vn; j++) {
                const e = Math.exp(w.logits[lo + j] - max);
                d[lo + j] = e;
                sum += e;
            }
            const inv = 1 / sum;
            const target = ids[offset + t + 1];
            loss += -Math.log(Math.max(d[lo + target] * inv, 1e-30));
            for (let j = 0; j < Vn; j++)
                d[lo + j] *= inv;
            d[lo + target] -= 1;
        }
        const invM = 1 / M;
        const n = M * Vn;
        for (let i = 0; i < n; i++)
            d[i] *= invM;
        return loss * invM;
    }
    /** Backward for the window the last forward() saw (uses w.dlogits). */
    backward(offset, M, ids = this.data) {
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
    zeroGrads() {
        for (const p of this.params)
            p.zero();
    }
    /** One forward+backward over a window, accumulating (unnormalised) gradients. */
    accumulate(offset) {
        const M = Math.min(this.T, this.data.length - 1 - offset);
        this.lastWindow = 0;
        if (M <= 0)
            return 0;
        const loss = this.forward(offset, M);
        this.backward(offset, M);
        this.lastWindow = M;
        return loss;
    }
    /** One AdamW step over a fresh random batch. */
    step() {
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
    adamW() {
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
    lossAt(offset) {
        const M = Math.min(this.T, this.data.length - 1 - offset);
        return this.forward(offset, M);
    }
    /** Fill gradients for a fixed window with the given divisor (gradient check). */
    gradsAt(offset, div = 1) {
        this.gradDiv = div;
        this.zeroGrads();
        const M = Math.min(this.T, this.data.length - 1 - offset);
        this.forward(offset, M);
        this.backward(offset, M);
        return M;
    }
    /* ---------------- generation ---------------- */
    #pick(temperature, topK) {
        const Vn = this.vocabSize;
        const base = (this.lastM - 1) * Vn;
        const temp = Math.max(temperature, 1e-4);
        const scores = new Float64Array(Vn);
        for (let j = 0; j < Vn; j++)
            scores[j] = this.w.logits[base + j] / temp;
        let pool;
        let max = -Infinity;
        if (topK > 0 && topK < Vn) {
            pool = Array.from({ length: Vn }, (_, j) => j)
                .sort((a, b) => scores[b] - scores[a])
                .slice(0, topK);
            for (const j of pool)
                if (scores[j] > max)
                    max = scores[j];
        }
        else {
            pool = Array.from({ length: Vn }, (_, j) => j);
            for (let j = 0; j < Vn; j++)
                if (scores[j] > max)
                    max = scores[j];
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
            if (r <= 0)
                return pool[i];
        }
        return pool[pool.length - 1];
    }
    /** Generate text. Returns the characters produced (the prompt is not included). */
    sample({ prompt = '', length = 240, temperature = 0.8, topK = 0, seed = null } = {}) {
        if (seed !== null)
            this.rand = mulberry32(seed);
        const T = this.T;
        const ids = prompt
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
            if (this.vocab.chars[next] === '\n' && ++newlines >= maxNewlines)
                break;
        }
        return decode(out.slice(ids.length), this.vocab);
    }
    /* ---------------- export ---------------- */
    exportWeights() {
        const params = {};
        for (const p of this.params)
            params[p.name] = Array.from(p.w, (x) => Number(x.toFixed(6)));
        return {
            name: 'llm-demo',
            architecture: 'character-level GPT: token + position embeddings, causal multi-head self-attention, feed-forward block, layer norm, residual connections',
            config: { ...this.config },
            vocab: this.vocab.chars,
            params,
        };
    }
    exportReport(extra = {}) {
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
