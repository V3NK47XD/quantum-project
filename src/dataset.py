"""Dataset loading, train/val/test splitting, and zero-leakage preprocessing."""

import os
import json
from pathlib import Path
from typing import Tuple, Dict, Any, List

import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.decomposition import PCA

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config
from src.features import extract_features_with_cache


PREPROCESSORS_PATH = config.MODELS_DIR / "preprocessors.joblib"


def load_dataset_features(
    use_cache: bool = True,
) -> Tuple[np.ndarray, np.ndarray, List[Dict[str, Any]]]:
    """Extract features for all audio files in bonafide and spoof directories."""
    bonafide_files = sorted(list(config.BONAFIDE_DIR.glob("*.flac")))
    spoof_files = sorted(list(config.SPOOF_DIR.glob("*.flac")))

    if not bonafide_files and not spoof_files:
        raise FileNotFoundError(
            f"No audio files found in {config.BONAFIDE_DIR} or {config.SPOOF_DIR}. "
            "Run data/download_asvspoof.py first."
        )

    features_list = []
    labels_list = []
    meta_list = []

    print(f"Loading {len(bonafide_files)} bonafide files...")
    for f in bonafide_files:
        feat, meta = extract_features_with_cache(f, use_cache=use_cache)
        features_list.append(feat)
        labels_list.append(config.LABEL_BONAFIDE)
        meta_list.append({
            "path": str(f),
            "filename": f.name,
            "label": config.LABEL_BONAFIDE,
            "class_name": config.LABEL_NAMES[config.LABEL_BONAFIDE],
            "hash": meta["hash"],
        })

    print(f"Loading {len(spoof_files)} spoof files...")
    for f in spoof_files:
        feat, meta = extract_features_with_cache(f, use_cache=use_cache)
        features_list.append(feat)
        labels_list.append(config.LABEL_SPOOF)
        meta_list.append({
            "path": str(f),
            "filename": f.name,
            "label": config.LABEL_SPOOF,
            "class_name": config.LABEL_NAMES[config.LABEL_SPOOF],
            "hash": meta["hash"],
        })

    X = np.array(features_list, dtype=np.float32)
    y = np.array(labels_list, dtype=np.int64)
    print(f"Extracted total dataset: X={X.shape}, y={y.shape}")
    return X, y, meta_list


def prepare_dataset_pipeline(
    X: np.ndarray,
    y: np.ndarray,
    test_size: float = config.TEST_SIZE,
    random_state: int = config.RANDOM_STATE,
    n_pca_components: int = config.PCA_COMPONENTS,
) -> Dict[str, Any]:
    """
    Split data and fit preprocessing pipeline strictly on training data
    to prevent train-test data leakage.
    """
    # 1. Stratified Train / Test split
    X_train_raw, X_test_raw, y_train, y_test = train_test_split(
        X, y,
        test_size=test_size,
        random_state=random_state,
        stratify=y,
    )

    # 2. Fit Classical Scaler ONLY on train data
    classical_scaler = StandardScaler()
    X_train_scaled = classical_scaler.fit_transform(X_train_raw)
    X_test_scaled = classical_scaler.transform(X_test_raw)

    # 3. Fit PCA strictly on train scaled features
    pca = PCA(n_components=n_pca_components, random_state=random_state)
    X_train_pca_unscaled = pca.fit_transform(X_train_scaled)
    X_test_pca_unscaled = pca.transform(X_test_scaled)

    # 4. Scale PCA features to [0, pi] for quantum angle rotation gates
    quantum_scaler = MinMaxScaler(feature_range=(0.0, np.pi))
    X_train_quantum = quantum_scaler.fit_transform(X_train_pca_unscaled)
    X_test_quantum = quantum_scaler.transform(X_test_pca_unscaled)

    # Calculate variance explained
    explained_var = pca.explained_variance_ratio_
    cumulative_var = np.sum(explained_var)
    print(f"PCA explained variance ratios: {explained_var.round(4)}, cumulative: {cumulative_var:.4f}")

    # Save preprocessing bundle
    pipeline = {
        "classical_scaler": classical_scaler,
        "pca": pca,
        "quantum_scaler": quantum_scaler,
        "n_pca_components": n_pca_components,
        "explained_variance_ratio": explained_var.tolist(),
        "cumulative_variance": float(cumulative_var),
        "feature_dim_raw": X.shape[1],
    }
    joblib.dump(pipeline, PREPROCESSORS_PATH)
    print(f"Saved preprocessing pipeline to {PREPROCESSORS_PATH}")

    return {
        "X_train_raw": X_train_raw,
        "X_test_raw": X_test_raw,
        "X_train_scaled": X_train_scaled,
        "X_test_scaled": X_test_scaled,
        "X_train_quantum": X_train_quantum,
        "X_test_quantum": X_test_quantum,
        "y_train": y_train,
        "y_test": y_test,
        "pipeline": pipeline,
    }


def preprocess_single_sample(
    feature_vector: np.ndarray,
    pipeline: Dict[str, Any] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Transform an unseen audio sample's feature vector for classical and quantum models."""
    if pipeline is None:
        if not PREPROCESSORS_PATH.exists():
            raise FileNotFoundError(f"Preprocessor not found at {PREPROCESSORS_PATH}. Train models first.")
        pipeline = joblib.load(PREPROCESSORS_PATH)

    feat_2d = feature_vector.reshape(1, -1)
    scaled_classical = pipeline["classical_scaler"].transform(feat_2d)
    pca_unscaled = pipeline["pca"].transform(scaled_classical)
    scaled_quantum = pipeline["quantum_scaler"].transform(pca_unscaled)

    return scaled_classical.astype(np.float32), scaled_quantum.astype(np.float32)
