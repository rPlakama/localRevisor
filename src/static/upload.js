(() => {
  const form = document.querySelector(".upload-form");
  const sourceInputs = document.querySelectorAll('input[name="source"]');
  const dropzone = document.getElementById("dropzone");
  const textField = document.getElementById("text-field");
  const textInput = document.getElementById("text");
  const fileInput = document.getElementById("file");
  const themeInput = document.getElementById("theme");
  const submitButton = document.getElementById("submit");
  const clientScoresInput = document.getElementById("client-scores");
  const clientExplanationInput = document.getElementById("client-explanation");
  const clientErrorsInput = document.getElementById("client-errors");
  const clientDescriptorModelInput = document.getElementById("client-descriptor-model");
  const wasmStatus = document.getElementById("wasm-status");

  if (
    !form ||
    !dropzone ||
    !textField ||
    !textInput ||
    !fileInput ||
    !themeInput ||
    !submitButton
  ) {
    return;
  }

  const clientWasmEnabled = form.dataset.clientWasm === "1";
  const localDescriptorEnabled = form.dataset.localDescriptor === "1";
  const localBaseUrl = form.dataset.localBaseUrl || "";
  const localModel = form.dataset.localModel || "";
  let wasmReadySubmitted = false;

  function isTextMode() {
    const checked = document.querySelector('input[name="source"]:checked');
    return checked ? checked.value === "text" : false;
  }

  function syncSubmit() {
    const hasContent = isTextMode()
      ? Boolean(textInput.value.trim())
      : Boolean(fileInput.files.length);
    submitButton.disabled = !hasContent || !themeInput.value.trim();
  }

  function syncSource() {
    dropzone.hidden = isTextMode();
    textField.hidden = !isTextMode();
    syncSubmit();
  }

  async function readEssayTextForWasm() {
    if (isTextMode()) {
      return textInput.value.trim();
    }
    const file = fileInput.files[0];
    if (!file) return "";
    const lower = (file.name || "").toLowerCase();
    if (lower.endsWith(".txt") || lower.endsWith(".md")) {
      return (await file.text()).trim();
    }
    return "";
  }

  function scoreWithWorker(text, theme) {
    return new Promise((resolve, reject) => {
      const worker = new Worker("/static/wasm_worker.js");
      const reqId = Date.now();
      worker.onmessage = (event) => {
        worker.terminate();
        const data = event.data || {};
        if (data.id === reqId && data.ok) {
          resolve(data);
        } else {
          reject(new Error(data.error || "Falha na inferência WebAssembly"));
        }
      };
      worker.onerror = (err) => {
        worker.terminate();
        reject(new Error(err.message || "Erro no Worker WebAssembly"));
      };
      worker.postMessage({
        id: reqId,
        text,
        theme,
        localDescriptor: localDescriptorEnabled,
        localBaseUrl,
        localModel,
      });
    });
  }

  if (clientWasmEnabled && clientScoresInput) {
    form.addEventListener("submit", async (event) => {
      if (wasmReadySubmitted) return;
      const essayText = await readEssayTextForWasm();
      if (!essayText) {
        return;
      }
      event.preventDefault();
      submitButton.disabled = true;
      if (wasmStatus) {
        wasmStatus.textContent = localDescriptorEnabled
          ? "Calculando notas e descritor no seu navegador..."
          : "Calculando notas C1–C5 no seu navegador via WebAssembly...";
      }
      try {
        const result = await scoreWithWorker(essayText, themeInput.value.trim());
        clientScoresInput.value = JSON.stringify(result.scores);
        if (result.descriptor && clientExplanationInput) {
          clientExplanationInput.value = result.descriptor.explanation || "";
          if (clientErrorsInput) {
            clientErrorsInput.value = result.descriptor.errors || "";
          }
          if (clientDescriptorModelInput) {
            clientDescriptorModelInput.value =
              result.descriptor.descriptorModel || "";
          }
        }
        wasmReadySubmitted = true;
        form.submit();
      } catch (err) {
        if (wasmStatus) {
          wasmStatus.textContent =
            "Aviso: inferência WebAssembly falhou; enviando normalmente.";
        }
        wasmReadySubmitted = true;
        form.submit();
      }
    });
  }

  sourceInputs.forEach((input) => input.addEventListener("change", syncSource));
  fileInput.addEventListener("change", syncSubmit);
  textInput.addEventListener("input", syncSubmit);
  themeInput.addEventListener("input", syncSubmit);
  syncSource();
})();
