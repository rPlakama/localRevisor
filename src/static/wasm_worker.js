/* Web Worker para inferência local no hardware do usuário via WebAssembly. */
"use strict";

const CONECTIVOS = [
  "portanto",
  "além disso",
  "entretanto",
  "contudo",
  "porém",
  "todavia",
  "no entanto",
  "dessa forma",
  "desse modo",
  "pois",
  "porque",
  "logo",
  "ademais",
  "sobretudo",
  "por isso",
  "embora",
  "visto que",
  "haja vista",
];

const PROPOSTA = [
  "deve",
  "devem",
  "cabe",
  "por meio de",
  "a fim de",
  "com o intuito de",
  "com o objetivo de",
  "para que",
];

let wasmPromise = null;
let weightsPromise = null;

function countOccurrences(haystack, needle) {
  let count = 0;
  let pos = 0;
  while ((pos = haystack.indexOf(needle, pos)) !== -1) {
    count += 1;
    pos += needle.length;
  }
  return count;
}

function extractFeatures(rawText) {
  const paragraphs = (rawText || "")
    .split(/\n+/)
    .map((p) => p.trim())
    .filter(Boolean);
  const paras = paragraphs.length > 0 ? paragraphs : [(rawText || "").trim()];
  const text = paras.join("\n");
  const lower = text.toLowerCase();
  const words = lower.match(/[\p{L}\p{N}_]+/gu) || [];
  const sentences = text
    .split(/[.!?]+/)
    .map((s) => s.trim())
    .filter(Boolean);
  const n = Math.max(words.length, 1);
  const uniqueWords = new Set(words).size;
  const connectives = CONECTIVOS.reduce(
    (acc, c) => acc + countOccurrences(lower, c),
    0
  );
  const commas = countOccurrences(text, ",");
  const lastLower = paras[paras.length - 1].toLowerCase();
  const proposal = PROPOSTA.reduce(
    (acc, p) => acc + countOccurrences(lastLower, p),
    0
  );

  return [
    words.length,
    paras.length,
    sentences.length,
    words.length / Math.max(sentences.length, 1),
    uniqueWords / n,
    (connectives / n) * 100,
    (commas / n) * 100,
    proposal,
  ];
}

async function getWasmInstance() {
  if (!wasmPromise) {
    wasmPromise = fetch("/static/model_score.wasm")
      .then((res) => {
        if (!res.ok) throw new Error("Falha ao carregar model_score.wasm");
        return res.arrayBuffer();
      })
      .then((bytes) => WebAssembly.instantiate(bytes))
      .then((result) => result.instance);
  }
  return wasmPromise;
}

async function getWeights() {
  if (!weightsPromise) {
    weightsPromise = fetch("/wasm-model/weights.json").then((res) => {
      if (!res.ok) throw new Error("Falha ao carregar pesos do modelo");
      return res.json();
    });
  }
  return weightsPromise;
}

function runWasmScore(instance, weights, features) {
  const memory = instance.exports.memory;
  const nFeat = features.length;
  const nComps = weights.comps.length;

  let offset = 0;
  const featPtr = offset;
  offset += nFeat * 8;
  const meanPtr = offset;
  offset += nFeat * 8;
  const scalePtr = offset;
  offset += nFeat * 8;
  const coefPtr = offset;
  offset += nComps * nFeat * 8;
  const interceptPtr = offset;
  offset += nComps * 8;
  const outPtr = offset;

  const f64 = new Float64Array(memory.buffer);
  f64.set(features, featPtr / 8);
  f64.set(weights.scaler_mean, meanPtr / 8);
  f64.set(weights.scaler_scale, scalePtr / 8);
  f64.set(weights.coef.flat(), coefPtr / 8);
  f64.set(weights.intercept, interceptPtr / 8);

  instance.exports.score(
    nFeat,
    nComps,
    featPtr,
    meanPtr,
    scalePtr,
    coefPtr,
    interceptPtr,
    outPtr
  );

  const out = new Int32Array(memory.buffer, outPtr, nComps);
  const scores = {};
  for (let i = 0; i < nComps; i += 1) {
    scores[weights.comps[i]] = out[i];
  }
  return scores;
}

self.onmessage = async (event) => {
  const { id, text } = event.data || {};
  try {
    const [instance, weights] = await Promise.all([
      getWasmInstance(),
      getWeights(),
    ]);
    const features = extractFeatures(text || "");
    const scores = runWasmScore(instance, weights, features);
    self.postMessage({ id, ok: true, scores });
  } catch (err) {
    self.postMessage({
      id,
      ok: false,
      error: err instanceof Error ? err.message : String(err),
    });
  }
};
