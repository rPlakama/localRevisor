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

function isLocalEndpoint(rawUrl) {
  try {
    const parsed = new URL(rawUrl);
    return (
      (parsed.protocol === "http:" || parsed.protocol === "https:") &&
      (parsed.hostname === "localhost" || parsed.hostname === "127.0.0.1")
    );
  } catch {
    return false;
  }
}

function buildOfflineDescriptor(theme, scores, features) {
  const [nWords, nParas, nSentences, wordsPerSentence, diversity, connectives, commas, proposal] = features;
  const total = Object.values(scores).reduce((a, b) => a + b, 0);
  const explanation = [
    `Avaliação gerada localmente no seu hardware (Descritor de Inferência Local) para o tema "${theme || "ENEM"}".`,
    `Nota total estimada: ${total}/1000 (C1: ${scores.C1}, C2: ${scores.C2}, C3: ${scores.C3}, C4: ${scores.C4}, C5: ${scores.C5}).`,
    `Estrutura observada: ${Math.round(nWords)} palavras, ${Math.round(nParas)} parágrafo(s) e ${Math.round(nSentences)} frase(s) (média de ${wordsPerSentence.toFixed(1)} palavras/frase).`,
    `Coesão e repertório lexical: diversidade vocabular de ${(diversity * 100).toFixed(1)}%, densidade de conectivos de ${connectives.toFixed(2)} por 100 palavras e ${commas.toFixed(2)} vírgulas por 100 palavras.`,
    proposal > 0
      ? `Proposta de intervenção: identificados ${Math.round(proposal)} marcador(es) de ação/intervenção no parágrafo final.`
      : "Proposta de intervenção: não foram detectados marcadores explícitos de intervenção (agente, ação, meio e finalidade) no parágrafo final.",
  ].join("\n");

  const warnings = [];
  if (nWords < 120) warnings.push("Texto curto: desenvolva mais os argumentos para fortalecer C2 e C3.");
  if (nParas < 3) warnings.push("Organize o texto em introdução, desenvolvimento e conclusão (mínimo 3–4 parágrafos).");
  if (connectives < 1.0) warnings.push("Amplie o uso de conectivos argumentativos (C4).");
  if (proposal === 0) warnings.push("Inclua agente, ação, meio/modo, finalidade e detalhamento na conclusão (C5).");

  return {
    explanation,
    errors: warnings.join("\n"),
    descriptorModel: "wasm-local-descriptor",
  };
}

async function tryLocalEndpointDescriptor(baseUrl, model, theme, text, scores) {
  if (!isLocalEndpoint(baseUrl) || !model) {
    return null;
  }
  const endpoint = `${baseUrl.replace(/\/+$/, "")}/chat/completions`;
  const scoreSummary = Object.entries(scores)
    .map(([k, v]) => `${k}: ${v}`)
    .join(", ");
  const resp = await fetch(endpoint, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
      Authorization: "Bearer local",
    },
    body: JSON.stringify({
      model,
      temperature: 0,
      messages: [
        {
          role: "system",
          content:
            "Você é um corretor de redações do ENEM. Responda em JSON com as chaves " +
            '{"explanation": "...", "errors": "..."} explicando as notas C1-C5 em português.',
        },
        {
          role: "user",
          content: `Tema: ${theme}\nNotas: ${scoreSummary}\n\nRedação:\n${text}`,
        },
      ],
    }),
  });
  if (!resp.ok) {
    return null;
  }
  const data = await resp.json();
  const content = data?.choices?.[0]?.message?.content || "";
  if (!content.trim()) {
    return null;
  }
  try {
    const parsed = JSON.parse(content);
    if (parsed && typeof parsed.explanation === "string") {
      return {
        explanation: parsed.explanation,
        errors: typeof parsed.errors === "string" ? parsed.errors : "",
        descriptorModel: model,
      };
    }
  } catch {
    // fallback para texto livre do modelo local
  }
  return {
    explanation: content.trim(),
    errors: "",
    descriptorModel: model,
  };
}

self.onmessage = async (event) => {
  const { id, text, theme, localDescriptor, localBaseUrl, localModel } = event.data || {};
  try {
    const [instance, weights] = await Promise.all([
      getWasmInstance(),
      getWeights(),
    ]);
    const features = extractFeatures(text || "");
    const scores = runWasmScore(instance, weights, features);
    let descriptor = null;
    if (localDescriptor) {
      try {
        descriptor = await tryLocalEndpointDescriptor(
          localBaseUrl || "",
          localModel || "",
          theme || "",
          text || "",
          scores
        );
      } catch {
        descriptor = null;
      }
      if (!descriptor) {
        descriptor = buildOfflineDescriptor(theme || "", scores, features);
      }
    }
    self.postMessage({ id, ok: true, scores, descriptor });
  } catch (err) {
    self.postMessage({
      id,
      ok: false,
      error: err instanceof Error ? err.message : String(err),
    });
  }
};
