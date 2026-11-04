"""Audio feature extraction module using Librosa with disk caching."""

import os
import hashlib
from pathlib import Path
from typing import Tuple, Dict, Any, Optional

import numpy as np
import librosa

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config


def get_audio_hash(audio_bytes_or_path: Any) -> str:
    """Generate SHA-256 hash for caching."""
    hasher = hashlib.sha256()
    if isinstance(audio_bytes_or_path, (str, Path)):
        path = Path(audio_bytes_or_path)
        if path.exists():
            with open(path, "rb") as f:
                while chunk := f.read(65536):
                    hasher.update(chunk)
        else:
            hasher.update(str(path).encode("utf-8"))
    elif isinstance(audio_bytes_or_path, bytes):
        hasher.update(audio_bytes_or_path)
    elif isinstance(audio_bytes_or_path, np.ndarray):
        hasher.update(audio_bytes_or_path.tobytes())
    else:
        hasher.update(str(audio_bytes_or_path).encode("utf-8"))
    return hasher.hexdigest()


def load_audio(
    source: Any,
    sr: int = config.SAMPLE_RATE,
    duration: float = config.DURATION,
) -> Tuple[np.ndarray, int]:
    """Load and normalize audio waveform from file path, bytes, or file-like object."""
    if isinstance(source, (str, Path)):
        y, orig_sr = librosa.load(str(source), sr=sr, mono=True)
    elif isinstance(source, bytes):
        import io
        import soundfile as sf
        try:
            y, orig_sr = sf.read(io.BytesIO(source))
        except Exception:
            import tempfile
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tf:
                tf.write(source)
                tmp_path = tf.name
            try:
                y, orig_sr = librosa.load(tmp_path, sr=sr, mono=True)
            finally:
                Path(tmp_path).unlink(missing_ok=True)
        if y.ndim > 1:
            y = np.mean(y, axis=1)
        if orig_sr != sr:
            y = librosa.resample(y, orig_sr=orig_sr, target_sr=sr)
    elif hasattr(source, "read"):
        source_bytes = source.read()
        return load_audio(source_bytes, sr=sr, duration=duration)
    elif isinstance(source, np.ndarray):
        y = source.astype(np.float32)
        if y.ndim > 1:
            y = np.mean(y, axis=1)
    else:
        raise ValueError(f"Unsupported audio source type: {type(source)}")

    # Fixed length padding or cropping
    target_samples = int(duration * sr)
    if len(y) < target_samples:
        y = np.pad(y, (0, target_samples - len(y)), mode="constant")
    elif len(y) > target_samples:
        y = y[:target_samples]

    # Normalize amplitude
    max_val = np.max(np.abs(y))
    if max_val > 1e-6:
        y = y / max_val

    return y.astype(np.float32), sr


def compute_features(
    y: np.ndarray,
    sr: int = config.SAMPLE_RATE,
) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
    """Compute MFCC, spectral centroid, bandwidth, zero-crossing rate, and chroma."""
    # 1. MFCC (mean & std across frames)
    mfcc = librosa.feature.mfcc(
        y=y,
        sr=sr,
        n_mfcc=config.N_MFCC,
        n_fft=config.N_FFT,
        hop_length=config.HOP_LENGTH,
    )
    mfcc_mean = np.mean(mfcc, axis=1)
    mfcc_std = np.std(mfcc, axis=1)

    # 2. Spectral Centroid (mean & std)
    centroid = librosa.feature.spectral_centroid(
        y=y,
        sr=sr,
        n_fft=config.N_FFT,
        hop_length=config.HOP_LENGTH,
    )
    centroid_mean = np.mean(centroid, axis=1)
    centroid_std = np.std(centroid, axis=1)

    # 3. Spectral Bandwidth (mean & std)
    bandwidth = librosa.feature.spectral_bandwidth(
        y=y,
        sr=sr,
        n_fft=config.N_FFT,
        hop_length=config.HOP_LENGTH,
    )
    bandwidth_mean = np.mean(bandwidth, axis=1)
    bandwidth_std = np.std(bandwidth, axis=1)

    # 4. Zero-Crossing Rate (mean & std)
    zcr = librosa.feature.zero_crossing_rate(
        y=y,
        hop_length=config.HOP_LENGTH,
    )
    zcr_mean = np.mean(zcr, axis=1)
    zcr_std = np.std(zcr, axis=1)

    # 5. Chroma STFT (mean & std)
    chroma = librosa.feature.chroma_stft(
        y=y,
        sr=sr,
        n_chroma=config.N_CHROMA,
        n_fft=config.N_FFT,
        hop_length=config.HOP_LENGTH,
    )
    chroma_mean = np.mean(chroma, axis=1)
    chroma_std = np.std(chroma, axis=1)

    # Concatenate into full feature vector
    feature_vector = np.concatenate([
        mfcc_mean,
        mfcc_std,
        centroid_mean,
        centroid_std,
        bandwidth_mean,
        bandwidth_std,
        zcr_mean,
        zcr_std,
        chroma_mean,
        chroma_std,
    ]).astype(np.float32)

    components = {
        "mfcc_mean": mfcc_mean,
        "mfcc_std": mfcc_std,
        "centroid_mean": centroid_mean,
        "centroid_std": centroid_std,
        "bandwidth_mean": bandwidth_mean,
        "bandwidth_std": bandwidth_std,
        "zcr_mean": zcr_mean,
        "zcr_std": zcr_std,
        "chroma_mean": chroma_mean,
        "chroma_std": chroma_std,
    }

    return feature_vector, components


def extract_features_with_cache(
    source: Any,
    use_cache: bool = True,
    sr: int = config.SAMPLE_RATE,
    duration: float = config.DURATION,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """Extract features with transparent disk caching."""
    cache_key = get_audio_hash(source)
    cache_path = config.CACHE_DIR / f"{cache_key}.npz"

    if use_cache and cache_path.exists():
        data = np.load(cache_path, allow_pickle=True)
        vec = data["feature_vector"]
        components = {k: data[k] for k in data.files if k != "feature_vector"}
        return vec, {"components": components, "cached": True, "hash": cache_key}

    # Load audio and compute
    y, sr = load_audio(source, sr=sr, duration=duration)
    vec, components = compute_features(y, sr=sr)

    if use_cache:
        save_dict = {"feature_vector": vec, **components}
        np.savez_compressed(cache_path, **save_dict)

    return vec, {"components": components, "cached": False, "hash": cache_key, "waveform": y}
