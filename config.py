"""Centralized configuration and hyperparameters for Quantum ML Voice Anti-Spoofing."""

import os
from pathlib import Path

# Paths
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
AUDIO_DIR = DATA_DIR / "audio"
BONAFIDE_DIR = AUDIO_DIR / "bonafide"
SPOOF_DIR = AUDIO_DIR / "spoof"
CACHE_DIR = DATA_DIR / "cache"
DEMO_DIR = DATA_DIR / "demo_samples"
MODELS_DIR = BASE_DIR / "models"
RESULTS_DIR = BASE_DIR / "results"

for d in [DATA_DIR, AUDIO_DIR, BONAFIDE_DIR, SPOOF_DIR, CACHE_DIR, DEMO_DIR, MODELS_DIR, RESULTS_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# Audio Feature Extraction Parameters
SAMPLE_RATE = 16000
DURATION = 3.0  # seconds (pad or crop audio to fixed length)
N_MFCC = 20
N_FFT = 1024
HOP_LENGTH = 512
N_CHROMA = 12

# Feature dimension after extraction:
# MFCC: 20 mean + 20 std = 40
# Centroid: 1 mean + 1 std = 2
# Bandwidth: 1 mean + 1 std = 2
# ZCR: 1 mean + 1 std = 2
# Chroma: 12 mean + 12 std = 24
# Total classical feature dimension = 70
TOTAL_FEATURE_DIM = 70

# Dataset Split
TEST_SIZE = 0.20
VAL_SIZE = 0.15
RANDOM_STATE = 42
TARGET_SUBSET_SIZE = 200  # Balanced subset (e.g. 100 bonafide + 100 spoof) for fast quantum training

# Quantum ML Parameters
N_QUBITS = 4
PCA_COMPONENTS = 4
FEATURE_MAP_TYPE = "ZZFeatureMap"  # or "AngleEmbedding"
ANSATZ_REPS = 2
QUANTUM_OPTIMIZER = "COBYLA"
QUANTUM_MAX_ITER = 60
QUANTUM_SHOTS = 1024
QUANTUM_SEED = 42

# Classical ML Parameters
CLASSICAL_EPOCHS = 40
CLASSICAL_BATCH_SIZE = 16
CLASSICAL_LR = 0.001
CLASSICAL_HIDDEN_1 = 64
CLASSICAL_HIDDEN_2 = 32
DROPOUT_RATE = 0.25

# Class Labels
LABEL_BONAFIDE = 0  # Human speech
LABEL_SPOOF = 1     # AI-generated / spoofed speech
LABEL_NAMES = {0: "Human (Bonafide)", 1: "AI-Generated (Spoof)"}
