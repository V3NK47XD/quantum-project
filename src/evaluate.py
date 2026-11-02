"""Evaluation and benchmarking module across CPU, GPU, and Quantum VQC."""

import json
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
)

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config


def compute_classification_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: Optional[np.ndarray] = None,
    training_time_sec: float = 0.0,
    inference_latency_ms: float = 0.0,
    model_name: str = "Model",
) -> Dict[str, Any]:
    """Calculate standard classification metrics and latency numbers."""
    y_true = np.asarray(y_true).astype(int)
    y_pred = np.asarray(y_pred).astype(int)

    acc = float(accuracy_score(y_true, y_pred))
    prec = float(precision_score(y_true, y_pred, zero_division=0))
    rec = float(recall_score(y_true, y_pred, zero_division=0))
    f1 = float(f1_score(y_true, y_pred, zero_division=0))
    cm = confusion_matrix(y_true, y_pred).tolist()

    # [[TN, FP], [FN, TP]]
    tn, fp, fn, tp = int(cm[0][0]), int(cm[0][1]), int(cm[1][0]), int(cm[1][1])

    report = classification_report(
        y_true,
        y_pred,
        target_names=[config.LABEL_NAMES[0], config.LABEL_NAMES[1]],
        output_dict=True,
        zero_division=0,
    )

    return {
        "model_name": model_name,
        "accuracy": acc,
        "precision": prec,
        "recall": rec,
        "f1_score": f1,
        "training_time_sec": float(training_time_sec),
        "inference_latency_ms": float(inference_latency_ms),
        "confusion_matrix": cm,
        "true_negatives": tn,
        "false_positives": fp,
        "false_negatives": fn,
        "true_positives": tp,
        "classification_report": report,
    }


def save_benchmark_results(results_list: List[Dict[str, Any]]) -> Tuple[Path, Path]:
    """Save benchmark results to both JSON and CSV files."""
    json_path = config.RESULTS_DIR / "evaluation_metrics.json"
    csv_path = config.RESULTS_DIR / "evaluation_metrics.csv"

    # Save JSON with full metadata
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(results_list, f, indent=2)

    # Save clean summary CSV for reporting
    rows = []
    for r in results_list:
        rows.append({
            "Model": r["model_name"],
            "Accuracy": round(r["accuracy"], 4),
            "Precision": round(r["precision"], 4),
            "Recall": round(r["recall"], 4),
            "F1-Score": round(r["f1_score"], 4),
            "Training Time (s)": round(r["training_time_sec"], 2),
            "Inference Latency (ms/sample)": round(r["inference_latency_ms"], 2),
            "True Positives (Spoof)": r["true_positives"],
            "False Positives (Spoof)": r["false_positives"],
            "True Negatives (Human)": r["true_negatives"],
            "False Negatives (Human)": r["false_negatives"],
        })

    df = pd.DataFrame(rows)
    df.to_csv(csv_path, index=False)
    print(f"Saved benchmark results to {json_path} and {csv_path}")
    return json_path, csv_path


def load_benchmark_results() -> Optional[List[Dict[str, Any]]]:
    """Load pre-computed benchmark results if available."""
    json_path = config.RESULTS_DIR / "evaluation_metrics.json"
    if json_path.exists():
        with open(json_path, "r", encoding="utf-8") as f:
            return json.load(f)
    return None
