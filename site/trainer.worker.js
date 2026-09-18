/**
 * trainer.worker.ts — runs the training and sampling off the main thread, so the
 * page keeps painting while the model learns.
 *
 * It is a module worker: it imports the same model code the page does. The page
 * falls back to running the identical host on the main thread if module workers
 * are unavailable (see app.ts) — the model code is the same either way.
 */
import { LlmHost } from './trainer-host.js';
// `lib` in tsconfig includes DOM (the page needs it) and not WebWorker, because
// the two cannot both be loaded — they declare `self` differently and conflict.
// This single cast is the price: everything below is typed from here on.
const ctx = self;
const host = new LlmHost((message) => {
    ctx.postMessage(message);
});
ctx.onmessage = (event) => {
    const request = event.data;
    switch (request.type) {
        case 'train':
            host.start(request);
            return;
        case 'stop':
            host.stop();
            return;
        case 'generate':
            host.generate(request);
            return;
        case 'export-weights':
            host.exportWeights();
            return;
        case 'export-report':
            host.exportReport(request.extra ?? {});
            return;
    }
    // Unreachable while HostRequest is exhaustive — and a compile error here is
    // exactly what should happen if a new request type is added and not handled.
    const unknown = request;
    ctx.postMessage({ type: 'error', message: `Unknown request: ${String(unknown.type)}` });
};
