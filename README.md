Como rodar:

 ```nix develop -c .venv/bin/flask --app src.main run --debug```

ou:

```docker compose up```

# localRevisor

Corretor de redações open source, com **modelo de IA configurável**: você escolhe qual IA usar (em nuvem ou local) e, opcionalmente, ativa um modelo treinado (BERTimbau) para estimar as notas por competência do Enem.

> O resultado é uma **estimativa para fins de estudo**. Ele não substitui a correção feita por professores nem a nota oficial.

## O que ele faz

- Recebe a redação por texto colado ou por arquivo (PDF, imagem, Markdown ou texto) e o tema.
- Devolve uma explicação e os pontos de melhoria, e guarda o histórico das correções.
- Funciona com qualquer serviço compatível com a API da OpenAI, inclusive modelos locais (por exemplo, com o Ollama).
- **Modo BERT (opcional):** um modelo BERTimbau, treinado com redações corrigidas por professores, estima as notas de C1 a C5. A IA escolhida só escreve a explicação de cada nota, citando trechos da redação.

## Requisitos

- Python 3.13 ou superior.
- Uma IA compatível com a API da OpenAI (chave de um provedor ou um servidor local, como o Ollama).

## Instalação

```
python -m venv .venv
```

Ative o ambiente virtual:

- Windows (PowerShell): `.venv\Scripts\activate`
- Linux e macOS: `source .venv/bin/activate`

Depois instale as dependências:

```
pip install -r requeriments.txt
```

## Executando

```
python -m flask --app src.main run --debug
```

Abra `http://127.0.0.1:5000`. Quem usa Nix pode rodar `nix develop -c .venv/bin/flask --app src.main run --debug`.

## Configurando a IA

Na tela de **Configurações**:

1. Informe a **URL base** do serviço (um endereço compatível com a API da OpenAI) e a **chave da API**.
2. Clique em **Conectar** e escolha o **modelo de correção** na lista.
3. Clique em **Salvar**.

A chave é gravada no arquivo `.env`, que o Git ignora. Nunca envie esse arquivo para o repositório.

**Usando um modelo local (Ollama):** instale o Ollama, baixe um modelo (`ollama pull NOME-DO-MODELO`) e use a URL base `http://localhost:11434/v1`. A chave pode ser qualquer texto, como `ollama`.

## Modo BERT (opcional)

Ele precisa de dependências a mais (o `torch` é grande) e dos arquivos do modelo treinado.

1. Instale as dependências do modo BERT:

   ```
   pip install -r requirements-bert.txt
   ```

   No Linux, para evitar baixar pacotes de GPU desnecessários, instale antes a versão para CPU do `torch`: `pip install torch==2.15.0 --index-url https://download.pytorch.org/whl/cpu`.

2. Baixe o modelo treinado: **[link do modelo: a definir]**. Extraia a pasta em `modelos/bertimbau_enem/`, que deve conter `config.json`, `model.safetensors`, `tokenizer.json` e `tokenizer_config.json`.

3. Inicie o app, abra as **Configurações** e marque a opção **"Usar o BERTimbau para estimar as notas"**. Clique em **Salvar**. A opção só fica disponível quando a pasta do modelo e as dependências estão presentes, e a própria tela explica o que falta. Desmarcada, o app funciona normalmente, só com a IA escolhida.

   Se preferir, a variável `USE_BERT=1` no arquivo `.env` define o valor inicial da opção na primeira execução. Depois disso, quem manda é a opção na tela de Configurações.

**Como funciona:** o BERT lê a redação e estima a nota de cada competência (de 0 a 200, em faixas de 40). A IA recebe essas notas e escreve a explicação de cada uma. A nota final é a soma das cinco competências.

**Limites conhecidos:**

- O modelo lê no máximo 512 "pedaços" de palavra. Redações mais longas são cortadas no final, e o app mostra um aviso.
- O modelo foi treinado com um corpus público de redações (Essay-BR). Temas muito diferentes dos do treino podem ter desempenho pior.
- A qualidade das explicações escritas pela IA depende do modelo escolhido e **pode conter imprecisões**.

## Estrutura do projeto

```
src/                   aplicação Flask (main.py, bert_notas.py, templates e arquivos estáticos)
modelos/               modelo BERT treinado (não vai para o Git)
data/                  banco de dados e arquivos enviados (não vai para o Git)
requeriments.txt       dependências do app
requirements-bert.txt  dependências do modo BERT (opcional)
```

## Docker

O repositório inclui `Dockerfile` e `docker-compose.yml`, mas eles **ainda não foram validados** nesta versão. A forma de execução testada é a instalação local descrita acima.

## Licença

A definir.