"""Multi-dataset ingestion orchestrator for voice anti-spoofing.

Combines:
1. ASVspoof 2019 LA benchmark (flac)
2. Kaggle human & nonhuman voices (mp3)
3. HuggingFace CommonVoice Indian Accent (mp3)
"""

import os
import io
import json
import shutil
from pathlib import Path
from typing import List, Dict, Any

import pyarrow.parquet as pq
import fsspec
import soundfile as sf

import sys
BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))
import config

try:
    import kagglehub
except ImportError:
    kagglehub = None


def ingest_asvspoof_if_needed(target_per_class: int = 100):
    """Ensure baseline ASVspoof files are in bonafide and spoof directories."""
    existing_bonafide = list(config.BONAFIDE_DIR.glob("LA_E_*.flac"))
    existing_spoof = list(config.SPOOF_DIR.glob("LA_E_*.flac"))
    if len(existing_bonafide) >= target_per_class and len(existing_spoof) >= target_per_class:
        print(f"[ASVspoof 2019] Already present: {len(existing_bonafide)} bonafide, {len(existing_spoof)} spoof.")
        return

    print("[ASVspoof 2019] Downloading/extracting shard...")
    from data.download_asvspoof import extract_balanced_subset
    extract_balanced_subset(target_per_class=target_per_class)


def ingest_kaggle_voices(
    num_human: int = 100,
    num_nonhuman: int = 200,
) -> Dict[str, int]:
    """Ingest human (bonafide) and nonhuman (spoof) audio from Kaggle dataset."""
    print(f"[Kaggle Voices] Sourcing {num_human} human and {num_nonhuman} nonhuman samples...")

    # Look for cached kagglehub download
    kaggle_path = None
    default_cache = Path.home() / ".cache" / "kagglehub" / "datasets" / "aabdurazzoq" / "human-and-nonhuman-voices"
    if default_cache.exists():
        version_dirs = sorted(list(default_cache.glob("versions/*")), reverse=True)
        if version_dirs:
            kaggle_path = version_dirs[0] / "human-nonhuman"
        else:
            candidates = list(default_cache.glob("**/human-nonhuman"))
            if candidates:
                kaggle_path = candidates[0]

    if kaggle_path is None or not kaggle_path.exists():
        if kagglehub is not None:
            print("[Kaggle Voices] Downloading dataset via kagglehub...")
            dl_path = kagglehub.dataset_download("aabdurazzoq/human-and-nonhuman-voices")
            kaggle_path = Path(dl_path) / "human-nonhuman"
        else:
            raise RuntimeError("kagglehub is not installed and Kaggle dataset not found locally.")

    human_src_dir = kaggle_path / "human"
    nonhuman_src_dir = kaggle_path / "nonhuman"

    human_files = sorted(list(human_src_dir.glob("*.mp3")))
    nonhuman_files = sorted(list(nonhuman_src_dir.glob("*.mp3")))

    print(f"[Kaggle Voices] Found {len(human_files)} human and {len(nonhuman_files)} nonhuman available files.")

    copied_human = 0
    for idx, f in enumerate(human_files[:num_human]):
        dest = config.BONAFIDE_DIR / f"kaggle_human_{idx+1:04d}_{f.name}"
        if not dest.exists():
            shutil.copy(f, dest)
        copied_human += 1

    copied_nonhuman = 0
    for idx, f in enumerate(nonhuman_files[:num_nonhuman]):
        dest = config.SPOOF_DIR / f"kaggle_nonhuman_{idx+1:04d}_{f.name}"
        if not dest.exists():
            shutil.copy(f, dest)
        copied_nonhuman += 1

    print(f"[Kaggle Voices] Ingested: {copied_human} human (bonafide), {copied_nonhuman} nonhuman (spoof).")
    return {"human": copied_human, "nonhuman": copied_nonhuman}


def ingest_indian_accent_voices(
    num_samples: int = 100,
) -> int:
    """Stream and ingest Indian accent human voices from Hugging Face dataset."""
    print(f"[HF Indian Accent] Sourcing {num_samples} Indian human voice samples...")
    url = (
        "https://huggingface.co/datasets/ishands/commonvoice-indian_accent/"
        "resolve/main/data/train-00000-of-00009.parquet"
    )

    # Check how many already ingested
    existing_indian = list(config.BONAFIDE_DIR.glob("indian_human_*.mp3"))
    if len(existing_indian) >= num_samples:
        print(f"[HF Indian Accent] Already ingested {len(existing_indian)} samples.")
        return len(existing_indian)

    fs = fsspec.filesystem("https")
    count = 0
    with fs.open(url, "rb") as pf_file:
        pf = pq.ParquetFile(pf_file)
        for batch in pf.iter_batches(batch_size=50, columns=["audio", "client_id", "accents", "sentence"]):
            df = batch.to_pandas()
            for _, row in df.iterrows():
                audio_dict = row["audio"]
                audio_bytes = audio_dict.get("bytes")
                if not audio_bytes:
                    continue

                client_id = str(row.get("client_id", "anon"))[:8]
                dest_path = config.BONAFIDE_DIR / f"indian_human_{count+1:04d}_{client_id}.mp3"
                if not dest_path.exists():
                    with open(dest_path, "wb") as f_out:
                        f_out.write(audio_bytes)

                count += 1
                if count >= num_samples:
                    break
            if count >= num_samples:
                break

    print(f"[HF Indian Accent] Ingested {count} Indian human voice samples into bonafide.")
    return count


def rebuild_unified_metadata() -> List[Dict[str, Any]]:
    """Build a unified metadata manifest for all audio files across all dataset sources."""
    metadata: List[Dict[str, Any]] = []

    # Bonafide directory
    for f in sorted(list(config.BONAFIDE_DIR.glob("*.*"))):
        if f.suffix.lower() not in [".flac", ".mp3", ".wav"]:
            continue

        source = "asvspoof_2019"
        if f.name.startswith("kaggle_human"):
            source = "kaggle_human_voices"
        elif f.name.startswith("indian_human"):
            source = "commonvoice_indian_accent"

        metadata.append({
            "utterance_id": f.stem,
            "filename": f.name,
            "file_path": str(f.relative_to(config.BASE_DIR)),
            "dataset_source": source,
            "label": config.LABEL_BONAFIDE,
            "label_name": config.LABEL_NAMES[config.LABEL_BONAFIDE],
        })

    # Spoof directory
    for f in sorted(list(config.SPOOF_DIR.glob("*.*"))):
        if f.suffix.lower() not in [".flac", ".mp3", ".wav"]:
            continue

        source = "asvspoof_2019"
        if f.name.startswith("kaggle_nonhuman"):
            source = "kaggle_nonhuman_voices"

        metadata.append({
            "utterance_id": f.stem,
            "filename": f.name,
            "file_path": str(f.relative_to(config.BASE_DIR)),
            "dataset_source": source,
            "label": config.LABEL_SPOOF,
            "label_name": config.LABEL_NAMES[config.LABEL_SPOOF],
        })

    meta_file = config.DATA_DIR / "metadata.json"
    with open(meta_file, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    bonafide_count = sum(1 for m in metadata if m["label"] == 0)
    spoof_count = sum(1 for m in metadata if m["label"] == 1)
    print(f"[Metadata] Unified catalog built: Total {len(metadata)} samples "
          f"({bonafide_count} Bonafide Human, {spoof_count} Spoof/Non-human).")

    # Refresh demo samples with representations from new sources
    config.DEMO_DIR.mkdir(parents=True, exist_ok=True)
    indian_samples = list(config.BONAFIDE_DIR.glob("indian_human_*.mp3"))
    if indian_samples:
        shutil.copy(indian_samples[0], config.DEMO_DIR / "demo_human_indian.mp3")

    kaggle_nonhuman = list(config.SPOOF_DIR.glob("kaggle_nonhuman_*.mp3"))
    if kaggle_nonhuman:
        shutil.copy(kaggle_nonhuman[0], config.DEMO_DIR / "demo_ai_kaggle.mp3")

    return metadata


def run_full_ingestion():
    print("=" * 60)
    print("  MULTI-DATASET INGESTION & BALANCING PIPELINE")
    print("=" * 60)
    ingest_asvspoof_if_needed(target_per_class=100)
    ingest_kaggle_voices(num_human=100, num_nonhuman=200)
    ingest_indian_accent_voices(num_samples=100)
    meta = rebuild_unified_metadata()
    print("[Complete] All datasets ingested and metadata manifest updated.")


if __name__ == "__main__":
    run_full_ingestion()
