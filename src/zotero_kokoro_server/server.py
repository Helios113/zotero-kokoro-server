from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional

import shutil
import tempfile

from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.responses import HTMLResponse, Response
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from .engine import KokoroEngine

_STATIC_DIR = Path(__file__).parent / "static"


class OpenAITTSRequest(BaseModel):
    model: Optional[str] = None
    input: str = Field(..., min_length=1)
    voice: str = Field(..., min_length=1)
    response_format: Optional[str] = "wav"
    speed: Optional[float] = None


def create_app(engine: KokoroEngine) -> FastAPI:
    app = FastAPI(
        title="Zotero Kokoro TTS",
        description="Local OpenAI-compatible TTS server powered by Kokoro, for use with Zotero Read Aloud.",
        version="0.1.0",
    )

    # Serve static assets (JS/CSS) if the directory exists and has files
    if _STATIC_DIR.exists() and any(_STATIC_DIR.iterdir()):
        app.mount("/static", StaticFiles(directory=_STATIC_DIR), name="static")

    # ------------------------------------------------------------------
    # UI
    # ------------------------------------------------------------------

    @app.get("/", response_class=HTMLResponse, include_in_schema=False)
    def ui() -> str:
        return _render_ui(engine)

    # ------------------------------------------------------------------
    # API
    # ------------------------------------------------------------------

    @app.get("/health")
    def health() -> Dict[str, str]:
        return {"status": "ok"}

    @app.get("/v1/voices")
    def v1_voices() -> List[Dict[str, str]]:
        return [{"id": v.id, "label": v.label, "locale": v.locale} for v in engine.list_voices()]

    @app.post("/v1/audio/speech")
    def v1_audio_speech(req: OpenAITTSRequest) -> Response:
        fmt = (req.response_format or "wav").lower()
        if fmt != "wav":
            raise HTTPException(status_code=400, detail=f"Unsupported response_format: {fmt}")
        try:
            wav = engine.synthesize_wav(text=req.input, voice=req.voice)
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))
        return Response(content=wav, media_type="audio/wav")

    @app.get("/api/stats")
    def api_stats() -> dict:
        return engine.stats()

    @app.get("/api/voices/cached")
    def api_voices_cached() -> List[str]:
        return engine.cached_voice_ids()

    @app.post("/api/cache/clear")
    def api_cache_clear() -> Dict[str, int]:
        n = engine.cache_clear()
        return {"cleared": n}

    @app.post("/api/voices/upload")
    async def api_voices_upload(file: UploadFile = File(...)) -> Dict[str, str]:
        if not (file.filename or "").endswith(".pt"):
            raise HTTPException(status_code=400, detail="Only .pt voice files are supported")
        voice_id = Path(file.filename).stem
        # Find or create the voices directory inside the HF snapshot cache
        snapshots = engine._repo_cache_dir() / "snapshots"
        if not snapshots.exists():
            raise HTTPException(
                status_code=503,
                detail="Kokoro model not yet downloaded. Run a synthesis first to trigger the download.",
            )
        # Use the most-recent snapshot dir
        snapshot_dirs = sorted(snapshots.iterdir())
        if not snapshot_dirs:
            raise HTTPException(status_code=503, detail="No snapshot found in HF cache.")
        voices_dir = snapshot_dirs[-1] / "voices"
        voices_dir.mkdir(exist_ok=True)
        dest = voices_dir / file.filename
        with dest.open("wb") as f:
            shutil.copyfileobj(file.file, f)
        return {"voice_id": voice_id, "path": str(dest)}

    return app


# ------------------------------------------------------------------
# Inline UI (single-file, no build step required)
# ------------------------------------------------------------------

def _render_ui(engine: KokoroEngine) -> str:
    return """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Kokoro TTS Server</title>
<style>
  *, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }
  :root {
    --bg: #0f1117; --surface: #1a1d27; --surface2: #22263a;
    --accent: #7c6fcd; --accent2: #a78bfa;
    --text: #e2e8f0; --muted: #64748b; --border: #2d3148;
    --green: #34d399; --red: #f87171; --yellow: #fbbf24;
    --radius: 10px; --gap: 16px;
  }
  body { background: var(--bg); color: var(--text); font-family: system-ui, sans-serif;
         font-size: 15px; line-height: 1.6; padding: var(--gap); min-height: 100vh; }
  h1 { font-size: 1.5rem; font-weight: 700; color: var(--accent2); margin-bottom: 4px; }
  h2 { font-size: 1rem; font-weight: 600; color: var(--muted); text-transform: uppercase;
       letter-spacing: .06em; margin-bottom: var(--gap); }
  .layout { display: grid; grid-template-columns: 1fr 1fr; gap: var(--gap); max-width: 1100px; margin: 0 auto; }
  @media (max-width: 700px) { .layout { grid-template-columns: 1fr; } }
  .card { background: var(--surface); border: 1px solid var(--border); border-radius: var(--radius); padding: var(--gap); }
  .card + .card { margin-top: 0; }
  .full { grid-column: 1 / -1; }
  label { display: block; font-size: .8rem; color: var(--muted); margin-bottom: 4px; margin-top: 12px; }
  label:first-child { margin-top: 0; }
  select, textarea, input[type=text], input[type=number] {
    width: 100%; background: var(--surface2); border: 1px solid var(--border);
    border-radius: 6px; color: var(--text); padding: 8px 10px; font-size: .9rem;
    outline: none; transition: border-color .15s;
  }
  select:focus, textarea:focus, input:focus { border-color: var(--accent); }
  textarea { resize: vertical; min-height: 90px; font-family: inherit; }
  button {
    display: inline-flex; align-items: center; gap: 6px; cursor: pointer;
    border: none; border-radius: 6px; padding: 8px 16px; font-size: .9rem;
    font-weight: 600; transition: opacity .15s, transform .1s;
  }
  button:active { transform: scale(.97); }
  button:disabled { opacity: .45; cursor: not-allowed; }
  .btn-primary { background: var(--accent); color: #fff; }
  .btn-primary:hover:not(:disabled) { opacity: .85; }
  .btn-ghost { background: var(--surface2); color: var(--text); border: 1px solid var(--border); }
  .btn-ghost:hover:not(:disabled) { border-color: var(--accent); }
  .btn-danger { background: #7f1d1d; color: #fca5a5; }
  .btn-danger:hover:not(:disabled) { opacity: .85; }
  .row { display: flex; gap: 8px; align-items: flex-end; flex-wrap: wrap; margin-top: 12px; }
  .pill { display: inline-block; padding: 2px 10px; border-radius: 999px; font-size: .75rem;
          font-weight: 600; background: var(--surface2); color: var(--muted); }
  .pill.ok { background: #064e3b; color: var(--green); }
  .pill.err { background: #7f1d1d; color: var(--red); }
  .stat-grid { display: grid; grid-template-columns: repeat(auto-fill, minmax(130px, 1fr)); gap: 10px; }
  .stat { background: var(--surface2); border-radius: 8px; padding: 12px; text-align: center; }
  .stat .val { font-size: 1.8rem; font-weight: 700; color: var(--accent2); }
  .stat .lbl { font-size: .72rem; color: var(--muted); margin-top: 2px; }
  .voice-grid { display: grid; grid-template-columns: repeat(auto-fill, minmax(200px, 1fr)); gap: 8px; }
  .voice-card {
    background: var(--surface2); border: 1px solid var(--border); border-radius: 8px;
    padding: 10px 12px; display: flex; align-items: center; justify-content: space-between; gap: 8px;
  }
  .voice-card .info { flex: 1; min-width: 0; }
  .voice-card .vid { font-weight: 600; font-size: .85rem; white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }
  .voice-card .vloc { font-size: .72rem; color: var(--muted); }
  .voice-card .cached-dot { width: 8px; height: 8px; border-radius: 50%; background: var(--muted); flex-shrink: 0; }
  .voice-card .cached-dot.cached { background: var(--green); }
  audio { width: 100%; margin-top: 10px; border-radius: 6px; }
  #status { font-size: .82rem; color: var(--muted); margin-top: 8px; min-height: 20px; }
  #status.err { color: var(--red); }
  .header { display: flex; align-items: center; justify-content: space-between;
             flex-wrap: wrap; gap: 8px; margin-bottom: var(--gap); max-width: 1100px; margin: 0 auto var(--gap); }
  .upload-area {
    border: 2px dashed var(--border); border-radius: var(--radius); padding: 24px;
    text-align: center; color: var(--muted); cursor: pointer; transition: border-color .2s;
  }
  .upload-area:hover, .upload-area.drag { border-color: var(--accent); color: var(--accent2); }
  .upload-area input { display: none; }
  .log { font-size: .78rem; font-family: monospace; background: var(--surface2); border-radius: 6px;
          padding: 10px; color: var(--muted); max-height: 120px; overflow-y: auto; white-space: pre-wrap; }
</style>
</head>
<body>

<div class="header">
  <div>
    <h1>🦜 Kokoro TTS Server</h1>
    <span id="health-badge" class="pill">checking…</span>
  </div>
  <button class="btn-ghost" onclick="refreshAll()">↻ Refresh</button>
</div>

<div class="layout">

  <!-- Playground -->
  <div class="card">
    <h2>Playground</h2>
    <label for="pg-text">Text</label>
    <textarea id="pg-text" placeholder="Type something to synthesize…">Hello! This is Kokoro TTS running locally.</textarea>
    <label for="pg-voice">Voice</label>
    <select id="pg-voice"></select>
    <div class="row">
      <button class="btn-primary" onclick="synthesize()" id="pg-btn">▶ Synthesize</button>
      <button class="btn-ghost" onclick="playRandom()">🎲 Random voice</button>
    </div>
    <div id="status"></div>
    <audio id="pg-audio" controls style="display:none"></audio>
  </div>

  <!-- Stats -->
  <div class="card">
    <h2>Stats</h2>
    <div class="stat-grid" id="stat-grid">
      <div class="stat"><div class="val" id="s-requests">—</div><div class="lbl">Requests</div></div>
      <div class="stat"><div class="val" id="s-hits">—</div><div class="lbl">Cache hits</div></div>
      <div class="stat"><div class="val" id="s-cache">—</div><div class="lbl">Cached items</div></div>
      <div class="stat"><div class="val" id="s-max">—</div><div class="lbl">Cache max</div></div>
    </div>
    <div class="row" style="margin-top:16px">
      <button class="btn-danger" onclick="clearCache()">🗑 Clear cache</button>
    </div>
    <div id="cache-log" class="log" style="margin-top:10px;display:none"></div>
  </div>

  <!-- Voices -->
  <div class="card full">
    <h2>Voices <span id="voice-count" class="pill">—</span></h2>
    <p style="font-size:.82rem;color:var(--muted);margin-bottom:12px">
      Green dot = weights cached locally. Grey = will download on first use.
    </p>
    <div class="voice-grid" id="voice-grid"></div>
  </div>

  <!-- Upload -->
  <div class="card full">
    <h2>Upload voice (.pt)</h2>
    <p style="font-size:.82rem;color:var(--muted);margin-bottom:12px">
      Drop a <code>.pt</code> voice file here to add it to the Kokoro voices cache.
      The file will be saved to <code>~/.cache/huggingface/…/voices/</code>.
    </p>
    <div class="upload-area" id="drop-zone" onclick="document.getElementById('file-input').click()"
         ondragover="event.preventDefault();this.classList.add('drag')"
         ondragleave="this.classList.remove('drag')"
         ondrop="handleDrop(event)">
      <input type="file" id="file-input" accept=".pt" onchange="handleFileInput(event)">
      <div>Drop <code>.pt</code> file here or click to browse</div>
    </div>
    <div id="upload-log" class="log" style="margin-top:10px;display:none"></div>
  </div>

</div>

<script>
let allVoices = [];
let cachedVoiceIds = new Set();

async function api(path, opts = {}) {
  const r = await fetch(path, opts);
  if (!r.ok) throw new Error(await r.text());
  return r;
}

async function checkHealth() {
  const el = document.getElementById('health-badge');
  try {
    await api('/health');
    el.textContent = '● online';
    el.className = 'pill ok';
  } catch {
    el.textContent = '● offline';
    el.className = 'pill err';
  }
}

async function loadVoices() {
  const [voicesRes, cachedRes] = await Promise.all([
    api('/v1/voices').then(r => r.json()),
    api('/api/voices/cached').then(r => r.json()),
  ]);
  allVoices = voicesRes;
  cachedVoiceIds = new Set(cachedRes);

  // Populate playground select
  const sel = document.getElementById('pg-voice');
  sel.innerHTML = allVoices.map(v =>
    `<option value="${v.id}">${v.label} (${v.locale})</option>`
  ).join('');

  // Populate voice grid
  document.getElementById('voice-count').textContent = allVoices.length;
  document.getElementById('voice-grid').innerHTML = allVoices.map(v => `
    <div class="voice-card">
      <div class="info">
        <div class="vid">${v.label}</div>
        <div class="vloc">${v.locale}</div>
      </div>
      <div class="cached-dot ${cachedVoiceIds.has(v.id) ? 'cached' : ''}" title="${cachedVoiceIds.has(v.id) ? 'Cached' : 'Not cached'}"></div>
      <button class="btn-ghost" style="padding:4px 10px;font-size:.78rem" onclick="previewVoice('${v.id}')">▶</button>
    </div>
  `).join('');
}

async function loadStats() {
  const s = await api('/api/stats').then(r => r.json());
  document.getElementById('s-requests').textContent = s.requests_total;
  document.getElementById('s-hits').textContent = s.cache_hits;
  document.getElementById('s-cache').textContent = s.cache_size;
  document.getElementById('s-max').textContent = s.cache_max;
}

function setStatus(msg, isErr = false) {
  const el = document.getElementById('status');
  el.textContent = msg;
  el.className = isErr ? 'err' : '';
}

async function synthesize(voiceOverride) {
  const text = document.getElementById('pg-text').value.trim();
  const voice = voiceOverride || document.getElementById('pg-voice').value;
  if (!text) { setStatus('Please enter some text.', true); return; }

  const btn = document.getElementById('pg-btn');
  btn.disabled = true;
  setStatus('Synthesizing…');

  try {
    const res = await api('/v1/audio/speech', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ model: 'kokoro', voice, input: text, response_format: 'wav' }),
    });
    const blob = await res.blob();
    const url = URL.createObjectURL(blob);
    const audio = document.getElementById('pg-audio');
    audio.src = url;
    audio.style.display = 'block';
    audio.play();
    setStatus(`Done — voice: ${voice}`);
    loadStats();
  } catch (e) {
    setStatus('Error: ' + e.message, true);
  } finally {
    btn.disabled = false;
  }
}

async function previewVoice(voiceId) {
  document.getElementById('pg-voice').value = voiceId;
  await synthesize(voiceId);
}

function playRandom() {
  if (!allVoices.length) return;
  const v = allVoices[Math.floor(Math.random() * allVoices.length)];
  document.getElementById('pg-voice').value = v.id;
  synthesize(v.id);
}

async function clearCache() {
  const log = document.getElementById('cache-log');
  log.style.display = 'block';
  try {
    const res = await api('/api/cache/clear', { method: 'POST' }).then(r => r.json());
    log.textContent = `Cleared ${res.cleared} cached items.`;
    loadStats();
  } catch (e) {
    log.textContent = 'Error: ' + e.message;
  }
}

function logUpload(msg) {
  const log = document.getElementById('upload-log');
  log.style.display = 'block';
  log.textContent += msg + '\\n';
}

async function uploadFile(file) {
  if (!file.name.endsWith('.pt')) {
    logUpload(`Skipped ${file.name}: not a .pt file`);
    return;
  }
  logUpload(`Uploading ${file.name}…`);
  const fd = new FormData();
  fd.append('file', file);
  try {
    const res = await fetch('/api/voices/upload', { method: 'POST', body: fd });
    const data = await res.json();
    if (!res.ok) throw new Error(data.detail || JSON.stringify(data));
    logUpload(`✓ ${file.name} → ${data.voice_id}`);
    await loadVoices();
  } catch (e) {
    logUpload(`✗ ${file.name}: ${e.message}`);
  }
}

function handleDrop(e) {
  e.preventDefault();
  document.getElementById('drop-zone').classList.remove('drag');
  [...e.dataTransfer.files].forEach(uploadFile);
}

function handleFileInput(e) {
  [...e.target.files].forEach(uploadFile);
}

async function refreshAll() {
  await Promise.all([checkHealth(), loadVoices(), loadStats()]);
}

// Boot
refreshAll();
</script>
</body>
</html>"""
