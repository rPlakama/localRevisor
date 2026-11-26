# DOCS — Pipeline Lógica do localRevisor

Este documento descreve, de forma visual e direta, como o **localRevisor** funciona de ponta a ponta: do envio da redação até a exibição da nota, passando por extração de texto, modo BERT (com janelas deslizantes), provedor de IA, validação JSON e persistência local.

---

## 1. Arquitetura em um olhar

```mermaid
flowchart LR
    U["Navegador (HTML / CSS / JS)"] <--> W["src/web/ (Flask + CSRF + Rotas)"]
    W <--> DB["src/db/ (SQLite em data/app.db)"]
    W --> UP["src/files.py (data/uploads/)"]
    W --> SEC["src/secrets_store.py (.env 600 / .secret)"]
    W --> WK["src/worker.py (Threads + Semáforo)"]
    WK --> GR["src/grading/ (Serviço, Parser e HTTP)"]
    GR --> BT["src/bert_notas/ (BERTimbau local opcional)"]
    GR <--> API["Provedor OpenAI-compatível (Nuvem ou Ollama)"]
    WK --> DB
```

---

## 2. Pipeline completa de uma correção

```mermaid
flowchart TD
    A["1. Envio em /upload (texto colado ou arquivo)"] --> B{"Tipo da entrada"}
    B -- "Texto colado" --> C["files.save_text() salva .txt UTF-8"]
    B -- "Arquivo" --> D["files.validate_and_store() valida extensão, tamanho e magic bytes"]
    C --> E["db.create_correction() grava status = corrigindo"]
    D --> E
    E --> F["worker.start() dispara thread respeitando limite de concorrência"]
    F --> G["grading.obtain_text() extrai o texto"]

    G --> G1{"Formato do arquivo"}
    G1 -- ".txt / .md" --> T["Leitura UTF-8 direta"]
    G1 -- ".pdf" --> P["pypdf extrai até 30 páginas"]
    P -- "PDF escaneado ou vazio" --> V["grading.transcribe() via modelo de visão"]
    P -- "Texto encontrado" --> T
    G1 -- "Imagem (.png/.jpg/.webp/.gif)" --> CV["Se .webp/.gif: ffmpeg converte para PNG"]
    CV --> V
    V --> T

    T --> M{"use_bert == 1?"}
    M -- "Não (modo llm)" --> LLM["Monta REVIEW_PROMPT (IA estima C1–C5 e explica)"]
    M -- "Sim (modo bert+llm)" --> BF{"bert_notas.motivo_indisponivel()"}
    BF -- "Indisponível (fail-closed)" --> ERR["db.fail_correction() status = erro"]
    BF -- "OK" --> BE["bert_notas.estimar() calcula C1–C5 em janelas de 512 tokens"]
    BE --> HYB["Monta HYBRID_PROMPT (IA apenas explica as notas fixas)"]

    LLM --> CALL["provider.create_completion() com fallback de parâmetros e retry em length"]
    HYB --> CALL

    CALL --> PARSE["parser.parse_review() ou parse_hybrid() + verify_quotes()"]
    PARSE -- "JSON inválido" --> REP["provider.repair_completion() (1 tentativa de reparo)"]
    REP -- "Falhou de novo" --> ERR
    REP -- "JSON válido" --> OK["db.finish_correction() status = corrigido"]
    PARSE -- "JSON válido" --> OK

    OK --> UI["/status notifica fim; /review/<id> exibe nota, C1–C5, explicação e erros"]
    ERR --> UI
```

---

## 3. Ciclo de vida de uma correção

```mermaid
stateDiagram-v2
    [*] --> corrigindo: POST /upload (cria registro + inicia worker)
    corrigindo --> corrigido: Sucesso na extração, avaliação e validação JSON
    corrigindo --> erro: Falha de arquivo, rede, provedor, BERT ou JSON
    corrigindo --> interrompido: Reinício do app (db.recover_interrupted)

    erro --> corrigindo: POST /correction/<id>/retry (Tentar novamente)
    interrompido --> corrigindo: POST /correction/<id>/retry (Tentar novamente)
    corrigido --> corrigindo: POST /correction/<id>/retry (Refazer correção)

    corrigido --> [*]: POST /correction/<id>/delete
    erro --> [*]: POST /correction/<id>/delete
    interrompido --> [*]: POST /correction/<id>/delete
```

> Enquanto existe qualquer correção em `corrigindo`, a página inicial faz polling leve em `GET /status` (`src/static/progress.js`) e recarrega automaticamente ao concluir.

---

## 4. Fluxo de configuração e descoberta de modelos (`/config`)

```mermaid
sequenceDiagram
    actor U as Usuário
    participant W as /config (routes_config.py)
    participant S as secrets_store.py (.env)
    participant DB as SQLite (settings)
    participant P as Provedor (/models)

    U->>W: Seleciona preset ou digita URL base + chave e clica em "Conectar"
    W->>S: Salva OPENAI_API_KEY em .env (permissão 600)
    W->>P: GET {base_url}/models (Authorization: Bearer ...)
    P-->>W: JSON com lista real de modelos
    W->>DB: Salva base_url e lista de modelos (sem preenchimento arbitrário)
    W-->>U: Renderiza <select> com os modelos retornados (+ opção "Outro")
    U->>W: Escolhe modelo de correção, visão e opções do BERT e clica em "Salvar"
    W->>DB: Persiste configuração final
```

---

## 5. Pipeline do BERTimbau para redações longas

O checkpoint ajustado do BERTimbau possui janela nativa fixa de **512 tokens** (`max_position_embeddings = 512`). Para não descartar a conclusão (onde normalmente está a proposta de intervenção / C5), `src/bert_notas/inference.py` aplica **janelamento deslizante**:

```mermaid
flowchart LR
    TXT["Texto da redação"] --> TOK["Tokenização (sem truncar)"]
    TOK --> CHK{"Total <= 510 tokens?"}
    CHK -- "Sim" --> S1["1 passagem direta (512 tokens)"]
    CHK -- "Não" --> LIM["Corta apenas se passar de bert_max_tokens (padrão 2048)"]
    LIM --> WIN["construir_janelas(): blocos de 510 tokens com overlap de 128; última janela ancorada no final"]
    WIN --> INF["Inferência BERT em cada janela ([CLS] + janela + [SEP])"]
    INF --> AVG["Média ponderada dos logits"]
    S1 --> NORM["Converte para 0–200 e arredonda em faixas de 40 (C1–C5)"]
    AVG --> NORM
```

- Se a redação cabe em `bert_max_tokens` (ajustável em `/config`: `512` a `8192`), o texto inteiro é lido e `truncated = False`.
- Se exceder `bert_max_tokens`, a revisão exibe o aviso indicando que é possível aumentar o limite em `/config`.

---

## 6. Mapa dos módulos (`src/`)

| Módulo / Pacote | Responsabilidade |
|---|---|
| `src/main.py` | Ponto de entrada Flask, carregamento de `.env`, `init_db()` e normalização inicial do BERT. |
| `src/web/__init__.py` | App factory `create_app()`. |
| `src/web/security.py` | Proteção CSRF, cabeçalhos de segurança (CSP, `X-Frame-Options`, etc.) e páginas 400/404/413/500. |
| `src/web/routes_corrections.py` | Rotas `/`, `/status`, `/upload`, `/review/<id>`, `/retry` e `/delete`. |
| `src/web/routes_config.py` | Rotas `/config` e `/config/baixar-bert`, presets de provedor e limite de tokens do BERT. |
| `src/web/feedback.py` | Formulário `/feedback`, validação, rate limit local e envio SMTP. |
| `src/worker.py` | Fila leve em threads com `BoundedSemaphore` (`LOCALREVISOR_MAX_WORKERS`). |
| `src/files.py` | Validação de upload (extensão, tamanho, UTF-8, magic bytes), leitura de PDF e conversão via `ffmpeg`. |
| `src/grading/models.py` | `ReviewResult`, `GradingError` e `ModelOutputError`. |
| `src/grading/prompts.py` | Prompts (`REVIEW_PROMPT`, `HYBRID_PROMPT`, `REPAIR_PROMPT`, `TRANSCRIBE_PROMPT`) e limites de tokens/timeout. |
| `src/grading/presets.py` | Catálogo de endpoints populares (OpenCode, Z.ai, Claude, Google AI, OpenAI, DeepSeek, OpenRouter, Groq, Ollama, LM Studio). |
| `src/grading/provider.py` | `GET /models`, cabeçalhos por provedor, fallback de parâmetros e retry automático em `finish_reason='length'`. |
| `src/grading/parser.py` | Extração de JSON, validação de notas ENEM (faixas de 40) e verificação de citações literais. |
| `src/grading/service.py` | Orquestração de `obtain_text()`, `transcribe()` e `review()`. |
| `src/bert_notas/validation.py` | Validação fail-closed do checkpoint (arquitetura, 5 saídas, `id2label` e dependências). |
| `src/bert_notas/inference.py` | Carregamento lazy com lock, janelas deslizantes (`construir_janelas`) e `estimar()`. |
| `src/bert_notas/download.py` | Download automático do checkpoint via `huggingface_hub.snapshot_download`. |
| `src/db/connection.py` | Conexão SQLite com `busy_timeout` e helpers de query/execução. |
| `src/db/schema.py` | Criação de tabelas, WAL e migrações idempotentes. |
| `src/db/settings.py` | Leitura e upsert da tabela `settings`. |
| `src/db/corrections.py` | CRUD de `corrections`, paginação, filtros, recuperação de jobs e serialização. |
| `src/paths.py` | Caminhos e limites configuráveis por variáveis de ambiente. |
| `src/secrets_store.py` | Gravação atômica da `OPENAI_API_KEY` em `.env` (`600`) e segredo de sessão em `data/.secret`. |

---

## 7. Garantias de segurança e resiliência

- **Local-first:** dados ficam em `data/app.db` (modo WAL) e `data/uploads/` (diretório `700`); chave de API nunca vai para o SQLite.
- **Proteções web:** CSRF obrigatório em todos os POSTs, `TRUSTED_HOSTS` restrito a loopback, CSP estrita (`script-src 'self' 'wasm-unsafe-eval'; style-src 'self'`) e validação de arquivos por magic bytes.
- **Fail-closed no BERT:** se o modo BERT estiver ligado sem checkpoint válido (e sem modo WebAssembly no cliente), a correção falha com mensagem clara em vez de inventar notas.
- **Recuperação automática:** jobs que estavam em `corrigindo` durante um reinício viram `interrompido` na inicialização e podem ser reexecutados com um clique.

---

## 8. Modo WebAssembly no cliente e Serviço NixOS

- **Inferência no hardware do usuário (`/config` → "Usar hardware (seu) para trabalhar com modelos"):** quando ativada, `src/static/upload.js` dispara `src/static/wasm_worker.js`, que carrega `src/static/model_score.wasm` e os pesos de `/wasm-model/weights.json` (`modelos/baseline_ridge/weights.json` ou pesos padrão), calcula `C1–C5` no navegador em WebAssembly e envia `client_scores` ao servidor — permitindo rodar o servidor hospedeiro sem PyTorch/BERT.
- **Pipeline de ML (`ml/pipeline.py`):** etapas automatizadas `collect` → `preprocess` → `train` → `evaluate` → `predict` (disponível também via `nix run .#ml-pipeline`).
- **Módulo NixOS (`nix/module.nix` / `nixosModules.default`):** expõe `services.localrevisor` com serviço `systemd` endurecido (`DynamicUser`, `StateDirectory`, `waitress`).
