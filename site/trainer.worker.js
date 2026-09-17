/**
 * trainer.worker.js — runs the training and sampling off the main thread, so
 * the page keeps painting while the model learns.
 *
 * It is a module worker: it imports the same llm.js the page does. The page
 * falls back to running the identical host on the main thread if module
 * workers are unavailable (see app.js) — the model code is the same either way.
 */

import { LlmHost } from './trainer-host.js';

const host = new LlmHost((message) => {
  self.postMessage(message);
});

self.onmessage = (event) => {
  const data = event.data || {};
  switch (data.type) {
    case 'train':
      host.start(data);
      break;
    case 'stop':
      host.stop();
      break;
    case 'generate':
      host.generate(data);
      break;
    case 'export-weights':
      host.exportWeights();
      break;
    case 'export-report':
      host.exportReport(data.extra || {});
      break;
    default:
      self.postMessage({ type: 'error', message: `Unknown request: ${data.type}` });
  }
};
