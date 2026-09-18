/**
 * app.ts — the page.
 *
 * Everything the visitor types stays in the browser: the corpus is handed to a
 * Worker on this origin, trained there, and never sent anywhere. There is no
 * fetch, no XHR, no beacon in this file or the model code, and a test asserts it
 * (test/static.test.js).
 *
 * 🔴 NO ASSET CARRIES A VERSION QUERY, AND `Cache-Control: no-store` ON THE WHOLE
 * SITE IS WHY. Cloudflare holds assets at the edge and in the browser for four
 * hours whatever the server sends, so on this site's first deploy the page loaded
 * the OLD llm.js and showed preset times that had already been corrected. The
 * first fix was a `?v=` token on every reference — but TypeScript cannot resolve
 * an import specifier with a query on it, so the durable fix is the header: the
 * Caddy block sets `no-store` for everything, and an end-to-end test asserts the
 * served JavaScript really arrives with it (test/e2e.test.js). Do not remove that
 * header, and do not reintroduce a build that needs a token bump to be visible.
 */
import { PRESETS, preset, parameterCount, buildVocab } from './llm.js';
import { CORPORA, DEFAULT_CORPUS } from './corpora.js';
/** Get an element the page is known to contain, or fail loudly. */
function $(id) {
    const el = document.getElementById(id);
    if (!el)
        throw new Error(`llm-demo: the page has no #${id}`);
    return el;
}
/* ------------------------------------------------------------------ *
 * Analytics — anonymous, only ever about the page, and only with consent
 * ------------------------------------------------------------------ *
 *
 * The page_view, element_click and scroll_depth events, and the tag itself, all
 * live in consent.ts, because that is the file that may not load Google's script
 * before the visitor says yes. This function is the whole of what this file
 * knows about it: it sends the events that are about the model — training
 * started, a model finished, text generated — and if no consent has been given
 * then `llmTrack` does not exist and nothing is sent at all.
 *
 * What is never sent, in any version of this file: the text the visitor pasted,
 * any sample the model produced, or any character of either. Only counts,
 * timings and preset names. */
function track(name, params = {}) {
    window.llmTrack?.(name, params);
}
function createWorkerTransport() {
    try {
        const worker = new Worker('./trainer.worker.js', { type: 'module' });
        let fellBack = false;
        return {
            send(request) {
                if (fellBack)
                    return;
                worker.postMessage(request);
            },
            onMessage(handler) {
                worker.onmessage = (event) => handler(event.data);
            },
            onError(handler) {
                worker.onerror = (event) => {
                    // A module worker can fail to load in an older browser. Rather than
                    // show a dead page, run the identical code here on the main thread.
                    if (fellBack)
                        return;
                    fellBack = true;
                    try {
                        worker.terminate();
                    }
                    catch {
                        /* already gone */
                    }
                    handler(event.message || 'the worker could not start');
                };
            },
        };
    }
    catch {
        return null;
    }
}
async function createLocalTransport() {
    const { LlmHost } = await import('./trainer-host.js');
    const host = new LlmHost((message) => queueMicrotask(() => currentHandler?.(message)));
    return {
        send(request) {
            switch (request.type) {
                case 'train': return host.start(request);
                case 'stop': return host.stop();
                case 'generate': return host.generate(request);
                case 'export-weights': return host.exportWeights();
                case 'export-report': return host.exportReport(request.extra ?? {});
            }
        },
        onMessage(handler) {
            currentHandler = handler;
        },
        onError() {
            /* the main thread cannot fail to load itself */
        },
    };
}
let currentHandler = null;
const state = {
    corpusKey: DEFAULT_CORPUS,
    presetKey: 'quick',
    training: false,
    loss: [],
    steps: 0,
    startedAt: 0,
    parameters: 0,
};
const number = (n) => n === null || n === undefined || !Number.isFinite(n) ? '—' : Math.round(n).toLocaleString();
const seconds = (s) => s < 60 ? `${s.toFixed(1)} s` : `${Math.floor(s / 60)} min ${Math.round(s % 60)} s`;
const roughSeconds = (s) => {
    if (s < 60)
        return `about ${s} seconds`;
    const minutes = Math.round(s / 60);
    return minutes === 1 ? 'about a minute' : `about ${minutes} minutes`;
};
/* ------------------------------------------------------------------ *
 * Corpus + preset controls
 * ------------------------------------------------------------------ */
function renderCorpora() {
    const box = $('corpusButtons');
    box.textContent = '';
    for (const key of Object.keys(CORPORA)) {
        const corpus = CORPORA[key];
        const button = document.createElement('button');
        button.type = 'button';
        button.className = 'chip';
        button.id = `corpus-${key}`;
        button.textContent = corpus.label;
        button.addEventListener('click', () => {
            state.corpusKey = key;
            $('corpus').value = corpus.text;
            renderCorpora();
            refreshCorpusStats();
            track('corpus_selected', { corpus: key });
        });
        if (key === state.corpusKey)
            button.classList.add('is-on');
        box.appendChild(button);
    }
    const own = document.createElement('button');
    own.type = 'button';
    own.className = 'chip' + (state.corpusKey === 'own' ? ' is-on' : '');
    own.id = 'corpus-own';
    own.textContent = 'Your own text';
    own.addEventListener('click', () => {
        state.corpusKey = 'own';
        $('corpus').focus();
        renderCorpora();
        refreshCorpusStats();
        track('corpus_selected', { corpus: 'own' });
    });
    box.appendChild(own);
}
function refreshCorpusStats() {
    const text = $('corpus').value;
    const vocab = buildVocab(text.length > 200000 ? text.slice(0, 200000) : text);
    const characters = [...text];
    $('corpusStats').textContent =
        `${characters.length.toLocaleString()} characters · ${vocab.size.toLocaleString()} distinct · ` +
            `${(text.split('\n').length - 1).toLocaleString()} lines`;
    const note = $('corpusNote');
    const corpus = CORPORA[state.corpusKey];
    note.textContent =
        state.corpusKey !== 'own' && corpus
            ? corpus.note
            : 'Paste anything — a page of prose, a list of product names, a script. Line structure is what a small model learns fastest.';
    refreshModelPreview();
}
function renderPresets() {
    const box = $('presetButtons');
    box.textContent = '';
    for (const key of Object.keys(PRESETS)) {
        const p = PRESETS[key];
        const button = document.createElement('button');
        button.type = 'button';
        button.className = 'chip' + (key === state.presetKey ? ' is-on' : '');
        button.id = `preset-${key}`;
        const strong = document.createElement('strong');
        strong.textContent = p.label;
        const small = document.createElement('span');
        small.textContent = `${roughSeconds(p.seconds)} on a desktop · ${p.nLayer} layer${p.nLayer > 1 ? 's' : ''}, ${p.nHead} heads`;
        button.append(strong, small);
        button.addEventListener('click', () => {
            state.presetKey = key;
            renderPresets();
            refreshModelPreview();
            track('preset_selected', { preset: key });
        });
        box.appendChild(button);
    }
}
function refreshModelPreview() {
    const config = preset(state.presetKey);
    const text = $('corpus').value;
    const vocab = buildVocab(text.length > 200000 ? text.slice(0, 200000) : text);
    state.parameters = parameterCount(config, vocab.size);
    const box = $('modelStats');
    box.textContent = '';
    const rows = [
        ['parameters', state.parameters.toLocaleString()],
        ['layers', String(config.nLayer)],
        ['attention heads', String(config.nHead)],
        ['width', String(config.dModel)],
        ['feed-forward', String(config.dFF)],
        ['context', `${config.blockSize} characters`],
        ['batch', `${config.batchSize} windows per step`],
        ['optimiser', `AdamW · learning rate ${config.lr}`],
        ['vocabulary', `${vocab.size} characters`],
    ];
    for (const [key, value] of rows) {
        const div = document.createElement('div');
        const dt = document.createElement('span');
        dt.textContent = key;
        const dd = document.createElement('b');
        dd.textContent = value;
        div.append(dt, dd);
        box.appendChild(div);
    }
    $('trainBtn').textContent = `Create the model — ${state.parameters.toLocaleString()} parameters`;
    // Showing the real parameter count is the point: it is a small model, and
    // pretending otherwise would be a lie a reader could check in ten seconds.
    $('paramAsides').textContent =
        `That is ${Math.round(state.parameters / 1000).toLocaleString()} thousand parameters. ` +
            `GPT-2 small is 124 million; a frontier model is far larger again. This one trains here, in your browser, in seconds.`;
}
/* ------------------------------------------------------------------ *
 * The loss chart
 * ------------------------------------------------------------------ */
function drawLoss() {
    const canvas = $('lossChart');
    const ctx = canvas.getContext('2d');
    if (!ctx)
        return;
    const ratio = window.devicePixelRatio || 1;
    const width = canvas.clientWidth || 320;
    const height = canvas.clientHeight || 90;
    canvas.width = Math.round(width * ratio);
    canvas.height = Math.round(height * ratio);
    ctx.setTransform(ratio, 0, 0, ratio, 0, 0);
    ctx.clearRect(0, 0, width, height);
    const values = state.loss;
    const styles = getComputedStyle(document.body);
    const line = styles.getPropertyValue('--accent').trim() || '#a78bfa';
    const grid = styles.getPropertyValue('--line').trim() || '#2a2340';
    ctx.strokeStyle = grid;
    ctx.lineWidth = 1;
    for (let i = 1; i < 4; i++) {
        const y = Math.round((height / 4) * i) + 0.5;
        ctx.beginPath();
        ctx.moveTo(0, y);
        ctx.lineTo(width, y);
        ctx.stroke();
    }
    if (values.length < 2) {
        // The waiting state used to be one dim line of text in an otherwise empty box,
        // and an empty box reads as a chart that is BROKEN rather than one that is
        // ready — George scrolled past it and asked where the training chart had gone.
        // So the empty state is now an invitation, and it says what will happen.
        ctx.fillStyle = styles.getPropertyValue('--muted').trim() || '#8b86a3';
        ctx.font = '600 13px system-ui, sans-serif';
        ctx.fillText('The loss curve is drawn here as it trains.', 12, height / 2 - 4);
        ctx.font = '12px system-ui, sans-serif';
        ctx.fillText('Press “Create the model” and watch the line fall.', 12, height / 2 + 17);
        return;
    }
    const max = Math.max(...values);
    const min = Math.min(...values);
    const span = Math.max(1e-6, max - min);
    const x = (i) => (i / (values.length - 1)) * width;
    const y = (v) => height - 4 - ((v - min) / span) * (height - 12);
    ctx.strokeStyle = line;
    ctx.lineWidth = 2;
    ctx.beginPath();
    values.forEach((v, i) => (i === 0 ? ctx.moveTo(x(i), y(v)) : ctx.lineTo(x(i), y(v))));
    ctx.stroke();
    // The end of the curve is marked and labelled, because the shape is the story and
    // the last point is where it has got to. The percentage is the drop from the
    // first loss, which is the number that means something to a reader with no
    // machine-learning background: how far it has fallen from where it started.
    const last = values[values.length - 1];
    ctx.fillStyle = line;
    ctx.beginPath();
    ctx.arc(x(values.length - 1) - 1.5, y(last), 2.8, 0, Math.PI * 2);
    ctx.fill();
    ctx.fillStyle = styles.getPropertyValue('--muted').trim() || '#8b86a3';
    ctx.font = '11px system-ui, sans-serif';
    ctx.fillText(max.toFixed(2), 4, 12);
    ctx.fillText(min.toFixed(2), 4, height - 4);
    const drop = values[0] > 0 ? Math.round((1 - last / values[0]) * 100) : 0;
    ctx.textAlign = 'right';
    ctx.fillText(`${last.toFixed(2)} — down ${drop}% from the start`, width - 6, 12);
    ctx.textAlign = 'left';
}
function setStatus(message, kind = '') {
    const el = $('statusMsg');
    el.textContent = message;
    el.className = 'status' + (kind ? ` is-${kind}` : '');
}
/* ------------------------------------------------------------------ *
 * Training
 * ------------------------------------------------------------------ */
let transport = null;
function startTraining() {
    const text = $('corpus').value;
    if (text.trim().length < 40) {
        setStatus('Give the model a little more text than that — at least a few lines.', 'warn');
        return;
    }
    state.training = true;
    state.loss = [];
    state.startedAt = performance.now();
    $('trainBtn').disabled = true;
    $('stopBtn').disabled = false;
    $('generateBtn').disabled = true;
    $('downloadWeights').disabled = true;
    $('downloadReport').disabled = true;
    $('samples').textContent = '';
    const fillEl = $('progressFill');
    fillEl.style.width = '0%';
    fillEl.parentElement?.setAttribute('aria-valuenow', '0');
    setStatus('Building the vocabulary and the first batch…');
    drawLoss();
    track('train_started', {
        preset: state.presetKey,
        corpus: state.corpusKey,
        characters: text.length,
        parameters: state.parameters,
    });
    transport?.send({ type: 'train', text, presetKey: state.presetKey });
}
function stopTraining() {
    transport?.send({ type: 'stop' });
    setStatus('Stopping after this batch…');
}
function onStarted(message) {
    state.steps = message.steps;
    $('configSummary').textContent =
        `${message.parameters.toLocaleString()} parameters · ${message.config.nLayer} layer(s) · ` +
            `${message.config.nHead} heads · width ${message.config.dModel} · context ${message.config.context} · ` +
            `${message.steps.toLocaleString()} steps · vocabulary ${message.vocabularySize}`;
    $('vocabChars').textContent = message.characters;
    setStatus(message.truncated
        ? 'That text was longer than the demo will use — the first 200,000 characters are being trained on.'
        : 'Training…', message.truncated ? 'warn' : '');
}
function onProgress(message) {
    state.loss.push(message.loss);
    const percent = Math.min(100, (message.step / message.steps) * 100);
    const fillEl = $('progressFill');
    fillEl.style.width = `${percent.toFixed(1)}%`;
    fillEl.parentElement?.setAttribute('aria-valuenow', String(Math.round(percent)));
    $('lossValue').textContent = message.loss.toFixed(3);
    $('stepValue').textContent = `${number(message.step)} / ${number(message.steps)}`;
    $('rateValue').textContent = `${Math.round(message.tokensPerSecond).toLocaleString()} tokens/s`;
    $('tokenValue').textContent = number(message.tokens);
    const remaining = message.stepsPerSecond > 0 ? (message.steps - message.step) / message.stepsPerSecond : 0;
    $('etaValue').textContent = remaining > 0.5 ? seconds(remaining) : 'almost done';
    drawLoss();
}
function onDone(message, stopped) {
    state.training = false;
    $('trainBtn').disabled = false;
    $('stopBtn').disabled = true;
    // A model exists from here on, so everything that acts on one becomes available.
    for (const id of ['generateBtn', 'downloadWeights', 'downloadReport']) {
        $(id).disabled = false;
    }
    const elapsed = (performance.now() - state.startedAt) / 1000;
    setStatus(stopped
        ? `Stopped at ${message.step.toLocaleString()} steps — the model is usable as it stands.`
        : `Trained ${message.step.toLocaleString()} steps in ${seconds(elapsed)}. Test it below.`, 'ok');
    if (!stopped) {
        track('model_trained', {
            preset: state.presetKey,
            corpus: state.corpusKey,
            steps: message.step,
            seconds: Math.round(elapsed),
            loss: Number((message.loss ?? 0).toFixed(3)),
            parameters: state.parameters,
        });
    }
    $('testPanel').classList.add('is-ready');
    $('testIntro').textContent =
        'These are samples from the model you just made — it has never seen any text other than what you gave it.';
    generateSamples();
}
/* ------------------------------------------------------------------ *
 * Generating
 * ------------------------------------------------------------------ */
function generateSamples() {
    $('samples').textContent = '';
    $('genMsg').textContent = 'Sampling from the model you just made…';
    track('text_generated', {
        temperature: Number($('temperature').value),
        top_k: Number($('topK').value),
        length: Number($('length').value),
    });
    transport?.send({
        type: 'generate',
        prompt: $('prompt').value,
        length: Number($('length').value),
        temperature: Number($('temperature').value),
        topK: Number($('topK').value),
        count: 3,
    });
}
function renderSample(message) {
    const block = document.createElement('pre');
    block.className = 'sample';
    block.textContent = message.text.trim() || '(the model produced nothing — try a lower temperature)';
    $('samples').appendChild(block);
}
/* ------------------------------------------------------------------ *
 * Downloads
 * ------------------------------------------------------------------ */
function download(name, text, type) {
    const blob = new Blob([text], { type });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = name;
    document.body.appendChild(a);
    a.click();
    a.remove();
    setTimeout(() => URL.revokeObjectURL(url), 4000);
}
function onWeights(payload) {
    const text = JSON.stringify(payload);
    download('llm-demo-weights.json', text, 'application/json');
    const megabytes = (text.length / 1048576).toFixed(1);
    setStatus(`Weights downloaded (${megabytes} MB) — ${payload.vocab.length} characters of vocabulary and every trained parameter, in plain JSON.`, 'ok');
    track('weights_downloaded', { parameters: state.parameters, megabytes: Number(megabytes) });
}
function onReport(payload) {
    const samples = [...document.querySelectorAll('#samples .sample')].map((el) => el.textContent ?? '');
    const lines = [
        '# Training report',
        '',
        `- generated: ${payload.generated_at}`,
        `- parameters: ${payload.model.parameters.toLocaleString()}`,
        `- architecture: ${payload.model.layers} layer(s), ${payload.model.heads} heads, width ${payload.model.width}, feed-forward ${payload.model.feed_forward}`,
        `- context: ${payload.model.context} characters · batch ${payload.model.batch_size}`,
        `- optimiser: ${payload.model.optimizer}, learning rate ${payload.model.learning_rate}, weight decay ${payload.model.weight_decay}`,
        `- tokenizer: ${payload.model.tokenizer}, vocabulary ${payload.model.vocabulary_size}`,
        `- corpus: ${payload.training.corpus_characters.toLocaleString()} characters`,
        `- trained: ${payload.training.steps.toLocaleString()} steps, ${payload.training.tokens_seen.toLocaleString()} tokens`,
        `- final loss: ${payload.training.smoothed_loss === null ? 'n/a' : payload.training.smoothed_loss.toFixed(4)}`,
        '',
        '## Loss curve (one point per recorded step)',
        '',
        ...payload.training.curve.map((v, i) => `- ${i + 1}: ${v.toFixed(4)}`),
        '',
        '## Samples taken from the finished model',
        '',
        ...samples.flatMap((s) => ['```', s, '```', '']),
        '## The full record',
        '',
        '```json',
        JSON.stringify(payload, null, 2),
        '```',
        '',
    ];
    download('llm-demo-training-report.md', lines.join('\n'), 'text/markdown');
    setStatus('Training report downloaded.', 'ok');
    track('report_downloaded', { steps: payload.training.steps });
}
/* ------------------------------------------------------------------ *
 * Wiring
 * ------------------------------------------------------------------ */
async function boot() {
    renderCorpora();
    renderPresets();
    $('corpus').value = CORPORA[DEFAULT_CORPUS].text;
    refreshCorpusStats();
    drawLoss();
    window.addEventListener('resize', drawLoss);
    transport = createWorkerTransport() ?? (await createLocalTransport());
    const active = transport;
    active.onError((message) => {
        setStatus(`Something went wrong in the trainer: ${message}`, 'warn');
        state.training = false;
        $('trainBtn').disabled = false;
        $('stopBtn').disabled = true;
    });
    active.onMessage((message) => {
        switch (message.type) {
            case 'started': return onStarted(message);
            case 'progress': return onProgress(message);
            case 'done': return onDone(message, false);
            case 'stopped': return onDone(message, true);
            case 'sample': return renderSample(message);
            case 'generated':
                $('genMsg').textContent = `Sampled ${message.count} times in ${message.seconds.toFixed(1)} s.`;
                return;
            case 'weights': return onWeights(message.payload);
            case 'report': return onReport(message.payload);
            case 'error': return setStatus(message.message, 'warn');
        }
    });
    $('trainBtn').addEventListener('click', startTraining);
    $('stopBtn').addEventListener('click', stopTraining);
    $('generateBtn').addEventListener('click', generateSamples);
    $('downloadWeights').addEventListener('click', () => active.send({ type: 'export-weights' }));
    $('downloadReport').addEventListener('click', () => active.send({
        type: 'export-report',
        extra: {
            corpus: state.corpusKey === 'own' ? 'pasted by the visitor' : CORPORA[state.corpusKey].label,
            preset: state.presetKey,
        },
    }));
    $('corpus').addEventListener('input', () => {
        if (state.corpusKey !== 'own') {
            state.corpusKey = 'own';
            renderCorpora();
        }
        refreshCorpusStats();
    });
    for (const id of ['temperature', 'topK', 'length']) {
        const input = $(id);
        input.addEventListener('input', () => {
            $(`${id}Value`).textContent = id === 'temperature' ? Number(input.value).toFixed(2) : input.value;
        });
    }
    // Nothing to start here: consent.ts owns analytics, and it has already decided
    // whether it is allowed to. See the note above `track`.
}
if (document.readyState === 'loading')
    document.addEventListener('DOMContentLoaded', () => void boot());
else
    void boot();
