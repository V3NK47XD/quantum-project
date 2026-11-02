"""Downloader and extractor for a balanced subset of ASVspoof 2019 LA dataset."""

import os
import json
import shutil
from pathlib import Path
import urllib.request
import pyarrow.parquet as pq

# Add project root to sys.path
import sys
BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))
import config

HF_PARQUET_URL = (
    "https://huggingface.co/datasets/SpeechAntiSpoofingBenchmarks/ASVspoof2019_LA/"
    "resolve/main/data/test-00000-of-00009.parquet"
)
TEMP_PARQUET_PATH = config.DATA_DIR / "temp_shard.parquet"
METADATA_PATH = config.DATA_DIR / "metadata.json"


def download_with_progress(url: str, dest: Path) -> None:
    print(f"Downloading shard from {url} to {dest}...")
    def reporthook(count, block_size, total_size):
        if count % 2000 == 0 and total_size > 0:
            percent = int(count * block_size * 100 / total_size)
            print(f"Downloaded {count * block_size / (1024*1024):.1f} MB / {total_size / (1024*1024):.1f} MB ({percent}%)")
    urllib.request.urlretrieve(url, dest, reporthook=reporthook)
    print(f"Download complete! File size: {dest.stat().st_size / (1024*1024):.2f} MB")


def extract_balanced_subset(target_per_class: int = 100) -> None:
    """Extract a balanced subset of bonafide and spoof audio files from parquet shard."""
    if not TEMP_PARQUET_PATH.exists():
        download_with_progress(HF_PARQUET_URL, TEMP_PARQUET_PATH)

    print("Reading parquet file...")
    table = pq.read_table(TEMP_PARQUET_PATH)
    print(f"Parquet table loaded: {table.num_rows} rows, schema: {table.column_names}")

    config.BONAFIDE_DIR.mkdir(parents=True, exist_ok=True)
    config.SPOOF_DIR.mkdir(parents=True, exist_ok=True)
    config.DEMO_DIR.mkdir(parents=True, exist_ok=True)

    metadata_records = []
    bonafide_count = 0
    spoof_count = 0

    p_col = table.column("path").to_pylist()
    l_col = table.column("label").to_pylist()
    a_col = table.column("audio").to_pylist()
    n_col = table.column("notes").to_pylist() if "notes" in table.column_names else [None] * len(l_col)

    print(f"Searching for {target_per_class} bonafide and {target_per_class} spoof samples...")
    for idx in range(table.num_rows):
        label = l_col[idx]
        # label can be 0 (bonafide) or 1 (spoof)
        if label == config.LABEL_BONAFIDE and bonafide_count < target_per_class:
            audio_data = a_col[idx]
            raw_bytes = audio_data.get("bytes") if isinstance(audio_data, dict) else None
            if not raw_bytes:
                continue

            orig_path = p_col[idx]
            filename = Path(orig_path).name if orig_path else f"bonafide_{bonafide_count:04d}.flac"
            out_file = config.BONAFIDE_DIR / filename
            with open(out_file, "wb") as f:
                f.write(raw_bytes)

            notes = json.loads(n_col[idx]) if (n_col[idx] and isinstance(n_col[idx], str)) else {}
            metadata_records.append({
                "utterance_id": notes.get("utterance_id", filename.replace(".flac", "")),
                "speaker_id": notes.get("speaker_id", "unknown"),
                "label": config.LABEL_BONAFIDE,
                "label_name": config.LABEL_NAMES[config.LABEL_BONAFIDE],
                "file_path": str(out_file.relative_to(config.BASE_DIR)),
                "filename": filename,
            })
            bonafide_count += 1

        elif label == config.LABEL_SPOOF and spoof_count < target_per_class:
            audio_data = a_col[idx]
            raw_bytes = audio_data.get("bytes") if isinstance(audio_data, dict) else None
            if not raw_bytes:
                continue

            orig_path = p_col[idx]
            filename = Path(orig_path).name if orig_path else f"spoof_{spoof_count:04d}.flac"
            out_file = config.SPOOF_DIR / filename
            with open(out_file, "wb") as f:
                f.write(raw_bytes)

            notes = json.loads(n_col[idx]) if (n_col[idx] and isinstance(n_col[idx], str)) else {}
            metadata_records.append({
                "utterance_id": notes.get("utterance_id", filename.replace(".flac", "")),
                "speaker_id": notes.get("speaker_id", "unknown"),
                "label": config.LABEL_SPOOF,
                "label_name": config.LABEL_NAMES[config.LABEL_SPOOF],
                "file_path": str(out_file.relative_to(config.BASE_DIR)),
                "filename": filename,
            })
            spoof_count += 1

        if bonafide_count >= target_per_class and spoof_count >= target_per_class:
            break

    print(f"Extracted: {bonafide_count} Bonafide (Human) and {spoof_count} Spoof (AI-Generated)")

    # Save metadata JSON
    with open(METADATA_PATH, "w", encoding="utf-8") as f:
        json.dump(metadata_records, f, indent=2)
    print(f"Saved metadata to {METADATA_PATH}")

    # Copy demo samples for Streamlit UI
    bonafide_files = list(config.BONAFIDE_DIR.glob("*.flac"))
    spoof_files = list(config.SPOOF_DIR.glob("*.flac"))

    for i in range(min(2, len(bonafide_files))):
        shutil.copy(bonafide_files[i], config.DEMO_DIR / f"demo_human_{i+1}.flac")
    for i in range(min(2, len(spoof_files))):
        shutil.copy(spoof_files[i], config.DEMO_DIR / f"demo_ai_spoof_{i+1}.flac")
    print(f"Copied demo samples to {config.DEMO_DIR}")

    # Clean up large parquet file to conserve disk space
    if TEMP_PARQUET_PATH.exists():
        TEMP_PARQUET_PATH.unlink()
        print(f"Removed temporary shard file {TEMP_PARQUET_PATH}")


if __name__ == "__main__":
    extract_balanced_subset(target_per_class=100)
