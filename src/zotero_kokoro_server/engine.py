from __future__ import annotations

import hashlib
import io
import threading
import wave
from dataclasses import dataclass, field
from glob import glob
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

try:
    from kokoro import KPipeline
except Exception as e:
    import sys
    if sys.version_info >= (3, 13):
        raise RuntimeError(
            "Kokoro currently requires Python 3.10–3.12. "
            "Create a venv with python3.11 or python3.12 and reinstall."
        ) from e
    raise RuntimeError("Failed to import kokoro. Install deps: pip install kokoro") from e


def _voice_lang_code(voice_id: str) -> str:
    return (voice_id or "").strip()[:1].lower() or "a"


def _wav_bytes(audio: np.ndarray, sample_rate: int = 24_000) -> bytes:
    audio = np.clip(np.asarray(audio, dtype=np.float32).reshape(-1), -1.0, 1.0)
    pcm16 = (audio * 32767.0).astype(np.int16)
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(pcm16.tobytes())
    return buf.getvalue()


DEFAULT_LOCALE: Dict[str, str] = {
    "a": "en-US", "b": "en-GB", "e": "es-ES", "f": "fr-FR",
    "h": "hi-IN", "i": "it-IT", "j": "ja-JP", "p": "pt-BR", "z": "zh-CN",
}

BASELINE_VOICES = ["af_heart", "af_bella", "af_nicole", "am_adam", "bf_emma"]


@dataclass(frozen=True)
class VoiceInfo:
    id: str
    label: str
    locale: str


@dataclass
class EngineStats:
    requests_total: int = 0
    cache_hits: int = 0
    cache_size: int = 0
    cache_max: int = 256


class KokoroEngine:
    def __init__(
        self,
        *,
        default_voice: str = "af_heart",
        repo_id: str = "hexgrad/Kokoro-82M",
        cache_max_entries: int = 256,
    ) -> None:
        self._default_voice = default_voice
        self._repo_id = repo_id
        self._pipelines: Dict[str, KPipeline] = {}
        self._pipelines_lock = threading.Lock()
        self._audio_cache: Dict[str, bytes] = {}
        self._audio_cache_lock = threading.Lock()
        self._cache_max = cache_max_entries
        self._stats = EngineStats(cache_max=cache_max_entries)
        self._stats_lock = threading.Lock()

    # ------------------------------------------------------------------
    # Voice catalog
    # ------------------------------------------------------------------

    def _hf_cache_root(self) -> Path:
        import os
        hf_home = os.environ.get("HF_HOME")
        return Path(hf_home) if hf_home else Path.home() / ".cache" / "huggingface"

    def _repo_cache_dir(self) -> Path:
        safe = self._repo_id.replace("/", "--")
        return self._hf_cache_root() / "hub" / f"models--{safe}"

    def list_voices(self) -> List[VoiceInfo]:
        cached: set[str] = set()
        snapshots = self._repo_cache_dir() / "snapshots"
        if snapshots.exists():
            for pt in glob(str(snapshots / "*" / "voices" / "*.pt")):
                cached.add(Path(pt).stem)

        voice_ids = sorted(set(BASELINE_VOICES) | cached)
        voices = []
        seen: set[str] = set()
        for vid in voice_ids:
            if vid in seen:
                continue
            seen.add(vid)
            lang = _voice_lang_code(vid)
            voices.append(VoiceInfo(
                id=vid,
                label=vid.replace("_", " "),
                locale=DEFAULT_LOCALE.get(lang, "en-US"),
            ))
        return voices

    def cached_voice_ids(self) -> List[str]:
        snapshots = self._repo_cache_dir() / "snapshots"
        if not snapshots.exists():
            return []
        return sorted({Path(pt).stem for pt in glob(str(snapshots / "*" / "voices" / "*.pt"))})

    # ------------------------------------------------------------------
    # Synthesis
    # ------------------------------------------------------------------

    def _get_pipeline(self, voice_id: str) -> KPipeline:
        lang_code = _voice_lang_code(voice_id)
        with self._pipelines_lock:
            if lang_code not in self._pipelines:
                self._pipelines[lang_code] = KPipeline(lang_code=lang_code, repo_id=self._repo_id)
            return self._pipelines[lang_code]

    def _cache_get(self, key: str) -> Optional[bytes]:
        with self._audio_cache_lock:
            return self._audio_cache.get(key)

    def _cache_put(self, key: str, value: bytes) -> None:
        with self._audio_cache_lock:
            if key not in self._audio_cache and len(self._audio_cache) >= self._cache_max:
                self._audio_cache.pop(next(iter(self._audio_cache)))
            self._audio_cache[key] = value

    def cache_clear(self) -> int:
        with self._audio_cache_lock:
            n = len(self._audio_cache)
            self._audio_cache.clear()
        return n

    def synthesize_wav(self, *, text: str, voice: str) -> bytes:
        voice = (voice or "").strip() or self._default_voice
        text = text.strip()
        if not text:
            raise ValueError("Empty input")

        cache_key = hashlib.sha256(f"{voice}\n{text}".encode()).hexdigest()

        with self._stats_lock:
            self._stats.requests_total += 1

        cached = self._cache_get(cache_key)
        if cached:
            with self._stats_lock:
                self._stats.cache_hits += 1
            return cached

        pipeline = self._get_pipeline(voice)
        chunks = [
            np.asarray(audio, dtype=np.float32)
            for _, _, audio in pipeline(text, voice=voice, speed=1, split_pattern=r"\n+")
        ]
        if not chunks:
            raise RuntimeError("No audio generated")

        wav_data = _wav_bytes(np.concatenate(chunks))
        self._cache_put(cache_key, wav_data)

        with self._stats_lock:
            self._stats.cache_size = len(self._audio_cache)

        return wav_data

    def stats(self) -> dict:
        with self._stats_lock:
            return {
                "requests_total": self._stats.requests_total,
                "cache_hits": self._stats.cache_hits,
                "cache_size": self._stats.cache_size,
                "cache_max": self._stats.cache_max,
                "repo_id": self._repo_id,
                "default_voice": self._default_voice,
            }
