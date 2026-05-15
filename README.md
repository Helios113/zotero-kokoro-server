# zotero-kokoro-server

A local [Kokoro TTS](https://github.com/hexgrad/kokoro) server with an OpenAI-compatible API and browser UI, designed for [Zotero](https://www.zotero.org) Read Aloud.

## Features

- **OpenAI-compatible API** — `POST /v1/audio/speech` and `GET /v1/voices`
- **Browser UI** at `http://localhost:8880` — voice playground, stats dashboard, voice upload
- **In-memory audio cache** — repeated requests are served instantly
- **Voice upload** — drop a `.pt` voice file in the UI to add new voices
- Works fully **offline** after initial model download

## Requirements

- Python 3.10–3.12 (Kokoro does not yet support 3.13+)

## Install

```bash
pip install zotero-kokoro-server
```

## Run

```bash
zotero-kokoro-server
```

Then open **http://localhost:8880** in your browser to access the UI.

Options:

```
--host          Bind address (default: 127.0.0.1, env: KOKORO_HOST)
--port          Port (default: 8880, env: KOKORO_PORT)
--default-voice Default voice ID (default: af_heart, env: KOKORO_DEFAULT_VOICE)
--repo-id       Hugging Face repo (default: hexgrad/Kokoro-82M, env: KOKORO_REPO_ID)
--cache-size    Max in-memory audio cache entries (default: 256)
```

The first run downloads model weights (~300MB) from Hugging Face. After that it runs offline. To pre-cache everything and then go fully offline:

```bash
# Run once online to cache model + SpaCy language model
zotero-kokoro-server

# Subsequent runs can use offline mode
HF_HUB_OFFLINE=1 zotero-kokoro-server
```

## Configure Zotero

In Zotero, open **Edit → Preferences → Advanced → Config Editor** and set:

| Preference | Value |
|---|---|
| `extensions.zotero.reader.readAloudLocal.enabled` | `true` |
| `extensions.zotero.reader.readAloudLocal.baseURL` | `http://127.0.0.1:8880` |
| `extensions.zotero.reader.readAloudLocal.protocol` | `openai` |
| `extensions.zotero.reader.readAloudLocal.openAIPath` | `/v1/audio/speech` |
| `extensions.zotero.reader.readAloudLocal.voicesPath` | `/v1/voices` |

Then open a PDF in the Reader and use the **Read Aloud** toolbar button.

## API

```bash
# Health check
curl http://127.0.0.1:8880/health

# List voices
curl http://127.0.0.1:8880/v1/voices

# Synthesize
curl -X POST http://127.0.0.1:8880/v1/audio/speech \
  -H 'Content-Type: application/json' \
  -d '{"voice":"af_heart","input":"Hello world","response_format":"wav"}' \
  --output out.wav

# Server stats
curl http://127.0.0.1:8880/api/stats

# Clear audio cache
curl -X POST http://127.0.0.1:8880/api/cache/clear
```

## License

MIT
