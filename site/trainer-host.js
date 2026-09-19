/**
 * trainer-host.ts — owns a Trainer and drives it in slices.
 *
 * The same logic runs in a Web Worker (the normal case, so the page stays
 * responsive) and on the main thread (the fallback, if module workers are not
 * available). It works in time-bounded slices rather than whole training runs so
 * that a "stop" message is always answered promptly.
 *
 * The request and reply types below are the contract across the worker boundary.
 * They are the reason this file is typed: a message the page reads must be a
 * message the host actually sends, and a typo in a field name is now a compile
 * error rather than an `undefined` at runtime.
 */
import { Trainer, buildVocab, encode, preset, parameterCount, plannedSteps } from './llm.js';
const SLICE_MS = 40;
const now = () => (typeof performance !== 'undefined' ? performance.now() : Date.now());
/* ------------------------------------------------------------------ *
 * The host
 * ------------------------------------------------------------------ */
export class LlmHost {
    emit;
    trainer = null;
    stopped = true;
    steps = 0;
    peakLoss = null;
    constructor(emit) {
        this.emit = emit;
    }
    get ready() {
        return this.trainer !== null;
    }
    start({ text = '', presetKey = 'quick', steps = null, seed = null, maxChars = 200000 }) {
        const trimmed = text.length > maxChars ? text.slice(0, maxChars) : text;
        if (trimmed.length < 40) {
            this.emit({ type: 'error', message: 'Give it a little more text than that — at least a few lines.' });
            return;
        }
        const vocab = buildVocab(trimmed);
        const data = encode(trimmed, vocab);
        // The preset's step count and time are resolved FOR THE LEVEL THE TEXT PICKED —
        // a character run needs many more steps than a word run to reach the same point.
        const config = preset(presetKey, vocab.level);
        if (steps)
            config.steps = Math.max(1, Math.floor(steps));
        const chosenSeed = seed === null || seed === undefined ? (Date.now() >>> 0) % 1e9 : seed;
        this.trainer = new Trainer({ data, vocab, config, seed: chosenSeed });
        this.stopped = false;
        this.peakLoss = null;
        // A small corpus cannot support an unbounded number of batches; `plannedSteps`
        // caps the run at MAX_EPOCHS passes so the progress bar and the estimate stay
        // truthful — and it is the same function the quality test uses, so the test
        // measures the run the page really performs.
        this.steps = plannedSteps(config, data.length);
        this.emit({
            type: 'started',
            seed: chosenSeed,
            steps: this.steps,
            corpusCharacters: trimmed.length,
            truncated: trimmed.length !== text.length,
            vocabularySize: vocab.size,
            vocabulary: vocab.level === 'char'
                ? vocab.chars.join('')
                : vocab.chars
                    .filter((c) => c !== '')
                    .map((c) => c.trim())
                    .join(' '),
            level: vocab.level,
            parameters: this.trainer.paramCount,
            config: { ...config, context: this.trainer.T, vocabularySize: vocab.size },
        });
        this.#schedule();
    }
    stop() {
        if (!this.trainer)
            return;
        this.stopped = true;
        this.emit({
            type: 'stopped',
            step: this.trainer.stepCount,
            steps: this.steps,
            loss: this.trainer.smoothedLoss ?? 0,
        });
    }
    #schedule() {
        setTimeout(() => this.#slice(), 0);
    }
    #slice() {
        const trainer = this.trainer;
        if (this.stopped || !trainer)
            return;
        const started = now();
        const step0 = trainer.stepCount;
        let last = { loss: 0, tokens: 0 };
        while (trainer.stepCount < this.steps && now() - started < SLICE_MS) {
            last = trainer.step();
        }
        const elapsed = (now() - started) / 1e3;
        const stepsDone = trainer.stepCount - step0;
        if (this.peakLoss === null || last.loss > this.peakLoss)
            this.peakLoss = last.loss;
        this.emit({
            type: 'progress',
            step: trainer.stepCount,
            steps: this.steps,
            loss: trainer.smoothedLoss ?? last.loss,
            rawLoss: last.loss,
            peakLoss: this.peakLoss,
            tokens: trainer.tokensSeen,
            tokensPerSecond: elapsed > 0 ? Math.round((last.tokens * stepsDone) / elapsed) : 0,
            stepsPerSecond: elapsed > 0 ? stepsDone / elapsed : 0,
        });
        if (trainer.stepCount >= this.steps) {
            this.emit({
                type: 'done',
                step: trainer.stepCount,
                steps: this.steps,
                loss: trainer.smoothedLoss ?? last.loss,
            });
            return;
        }
        this.#schedule();
    }
    /** Generate `count` samples; one message per sample so the page fills in live. */
    generate({ prompt = '', length = 200, temperature = 0.8, topK = 0, count = 3, seed = null }) {
        const trainer = this.trainer;
        if (!trainer) {
            this.emit({ type: 'error', message: 'Train the model first — then there is something to test.' });
            return;
        }
        const baseSeed = seed === null || seed === undefined ? (Date.now() >>> 0) % 1e9 : seed;
        const started = now();
        for (let i = 0; i < count; i++) {
            const text = trainer.sample({ prompt, length, temperature, topK, seed: baseSeed + i * 7919 });
            this.emit({ type: 'sample', index: i, count, text, prompt });
        }
        this.emit({ type: 'generated', count, seconds: (now() - started) / 1e3 });
    }
    exportWeights() {
        if (!this.trainer) {
            this.emit({ type: 'error', message: 'There is no model to export yet.' });
            return;
        }
        this.emit({ type: 'weights', payload: this.trainer.exportWeights() });
    }
    exportReport(extra = {}) {
        if (!this.trainer) {
            this.emit({ type: 'error', message: 'There is no model to report on yet.' });
            return;
        }
        this.emit({ type: 'report', payload: this.trainer.exportReport(extra) });
    }
    /** Parameter count for a preset and corpus, before any training. */
    static preview(presetKey, text) {
        const config = preset(presetKey);
        const vocab = buildVocab(text.length > 200000 ? text.slice(0, 200000) : text);
        return { parameters: parameterCount(config, vocab.size), vocabularySize: vocab.size };
    }
}
