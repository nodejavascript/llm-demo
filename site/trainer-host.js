/**
 * trainer-host.js — owns a Trainer and drives it in slices.
 *
 * The same logic runs in a Web Worker (the normal case, so the page stays
 * responsive) and on the main thread (the fallback, if module workers are not
 * available). It works in time-bounded slices rather than whole training runs
 * so that a "stop" message is always answered promptly.
 */

import { Trainer, buildVocab, encode, preset, parameterCount } from './llm.js?v=2';

const SLICE_MS = 40;
const now = () => (typeof performance !== 'undefined' ? performance.now() : Date.now());

export class LlmHost {
  /**
   * @param {(message: object) => void} emit  called with every message the
   *        transport should carry back to the page
   */
  constructor(emit) {
    this.emit = emit;
    this.trainer = null;
    this.stopped = true;
    this.config = null;
    this.running = false;
    this.peakLoss = null;
  }

  start({ text = '', presetKey = 'quick', steps = null, seed = null, maxChars = 200000 }) {
    const trimmed = text.length > maxChars ? text.slice(0, maxChars) : text;
    if (trimmed.length < 40) {
      this.emit({ type: 'error', message: 'Give it a little more text than that — at least a few lines.' });
      return;
    }
    const vocab = buildVocab(trimmed);
    const data = encode(trimmed, vocab);
    const config = preset(presetKey);
    if (steps) config.steps = Math.max(1, Math.floor(steps));
    const chosenSeed = seed === null || seed === undefined ? (Date.now() >>> 0) % 1e9 : seed;

    this.trainer = new Trainer({ data, vocab, config, seed: chosenSeed });
    this.config = config;
    this.stopped = false;
    this.running = true;
    this.peakLoss = null;

    const effectiveSteps = Math.min(config.steps, Math.max(1, Math.floor((data.length - 1) * 40 / (config.batchSize * config.blockSize))));
    this.steps = effectiveSteps;

    this.emit({
      type: 'started',
      seed: chosenSeed,
      steps: effectiveSteps,
      corpusCharacters: trimmed.length,
      truncated: trimmed.length !== text.length,
      vocabularySize: vocab.size,
      characters: vocab.chars.filter((c) => c !== '\uFFFD').join(''),
      parameters: this.trainer.paramCount,
      config: {
        ...config,
        context: this.trainer.T,
        vocabularySize: vocab.size,
      },
    });
    this.#schedule();
  }

  stop() {
    if (!this.trainer) return;
    this.stopped = true;
    this.running = false;
    this.emit({ type: 'stopped', step: this.trainer.stepCount, steps: this.steps });
  }

  #schedule() {
    setTimeout(() => this.#slice(), 0);
  }

  #slice() {
    if (this.stopped || !this.trainer) return;
    const started = now();
    const step0 = this.trainer.stepCount;
    let last = 0;
    while (
      this.trainer.stepCount < this.steps &&
      now() - started < SLICE_MS
    ) {
      last = this.trainer.step();
    }
    const elapsed = (now() - started) / 1e3;
    const stepsDone = this.trainer.stepCount - step0;
    if (this.peakLoss === null || last.loss > this.peakLoss) this.peakLoss = last.loss;

    const smoothed = this.trainer.smoothedLoss;
    this.emit({
      type: 'progress',
      step: this.trainer.stepCount,
      steps: this.steps,
      loss: smoothed,
      rawLoss: last.loss,
      peakLoss: this.peakLoss,
      tokens: this.trainer.tokensSeen,
      tokensPerSecond: elapsed > 0 ? Math.round(last.tokens * stepsDone / elapsed) : 0,
      stepsPerSecond: elapsed > 0 ? stepsDone / elapsed : 0,
    });

    if (this.trainer.stepCount >= this.steps) {
      this.running = false;
      this.emit({ type: 'done', step: this.trainer.stepCount, steps: this.steps, loss: this.trainer.smoothedLoss });
      return;
    }
    this.#schedule();
  }

  /** Generate `count` samples; one message per sample so the page fills in live. */
  generate({ prompt = '', length = 200, temperature = 0.8, topK = 0, count = 3, seed = null }) {
    if (!this.trainer) {
      this.emit({ type: 'error', message: 'Train the model first — then there is something to test.' });
      return;
    }
    const baseSeed = seed === null || seed === undefined ? (Date.now() >>> 0) % 1e9 : seed;
    const started = now();
    for (let i = 0; i < count; i++) {
      const text = this.trainer.sample({ prompt, length, temperature, topK, seed: baseSeed + i * 7919 });
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

  /** Parameter count for the currently selected preset and corpus, before any training. */
  static preview(presetKey, text) {
    const config = preset(presetKey);
    const vocab = buildVocab(text.length > 200000 ? text.slice(0, 200000) : text);
    return { parameters: parameterCount(config, vocab.size), vocabularySize: vocab.size };
  }
}
