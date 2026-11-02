"""Visualization utilities for audio waveforms, spectrograms, circuits, and metrics."""

from typing import Dict, Any, List, Optional
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend for server/Streamlit
import librosa
import librosa.display

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config


def plot_confusion_matrix(cm: List[List[int]], model_name: str = "Model") -> plt.Figure:
    """Plot an annotated confusion matrix heatmap."""
    fig, ax = plt.subplots(figsize=(4.5, 3.8), dpi=150)
    cm_arr = np.array(cm)

    im = ax.imshow(cm_arr, interpolation="nearest", cmap=plt.cm.Blues)
    ax.figure.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    classes = ["Human", "AI Spoof"]
    tick_marks = np.arange(len(classes))
    ax.set_xticks(tick_marks)
    ax.set_xticklabels(classes, fontsize=10)
    ax.set_yticks(tick_marks)
    ax.set_yticklabels(classes, fontsize=10)

    # Values inside boxes
    thresh = cm_arr.max() / 2.0 if cm_arr.max() > 0 else 1.0
    for i in range(cm_arr.shape[0]):
        for j in range(cm_arr.shape[1]):
            val = cm_arr[i, j]
            color = "white" if val > thresh else "black"
            ax.text(j, i, f"{val}", ha="center", va="center", color=color, fontsize=12, fontweight="bold")

    ax.set_ylabel("True Ground Truth", fontsize=11, fontweight="bold")
    ax.set_xlabel("Model Prediction", fontsize=11, fontweight="bold")
    ax.set_title(f"Confusion Matrix: {model_name}", fontsize=12, pad=10)
    fig.tight_layout()
    return fig


def plot_metrics_comparison(benchmark_results: List[Dict[str, Any]]) -> plt.Figure:
    """Plot comparative grouped bar chart for Accuracy, Precision, Recall, F1."""
    fig, ax = plt.subplots(figsize=(8.0, 4.2), dpi=150)

    models = [r["model_name"] for r in benchmark_results]
    metrics = ["Accuracy", "Precision", "Recall", "F1-Score"]
    metric_keys = ["accuracy", "precision", "recall", "f1_score"]

    x = np.arange(len(models))
    width = 0.18
    colors = ["#2b5c8f", "#3c9d9b", "#f4a261", "#e76f51"]

    for i, (m_name, m_key) in enumerate(zip(metrics, metric_keys)):
        values = [r[m_key] for r in benchmark_results]
        bars = ax.bar(x + (i - 1.5) * width, values, width, label=m_name, color=colors[i], alpha=0.9, edgecolor="black", linewidth=0.6)
        for bar in bars:
            height = bar.get_height()
            ax.annotate(f"{height:.2f}",
                        xy=(bar.get_x() + bar.get_width() / 2, height),
                        xytext=(0, 3), textcoords="offset points",
                        ha="center", va="bottom", fontsize=8, fontweight="bold")

    ax.set_ylabel("Score (0.0 to 1.0)", fontsize=11, fontweight="bold")
    ax.set_title("Performance Metric Comparison (Identical Held-Out Test Data)", fontsize=13, pad=12)
    ax.set_xticks(x)
    ax.set_xticklabels(models, fontsize=10, fontweight="bold")
    ax.set_ylim(0.0, 1.15)
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    ax.legend(loc="lower right", framealpha=0.9)
    fig.tight_layout()
    return fig


def plot_latency_and_training_comparison(benchmark_results: List[Dict[str, Any]]) -> plt.Figure:
    """Plot training time and inference latency across CPU, GPU, and Quantum."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9.0, 3.8), dpi=150)

    models = [r["model_name"] for r in benchmark_results]
    train_times = [r["training_time_sec"] for r in benchmark_results]
    infer_latencies = [r["inference_latency_ms"] for r in benchmark_results]

    x = np.arange(len(models))
    colors = ["#348AA7", "#525174", "#513B56"]

    # Training Time
    b1 = ax1.bar(x, train_times, color=colors, width=0.55, edgecolor="black", linewidth=0.6)
    ax1.set_ylabel("Total Training Time (seconds)", fontsize=10, fontweight="bold")
    ax1.set_title("Model Training Time", fontsize=11, fontweight="bold")
    ax1.set_xticks(x)
    ax1.set_xticklabels(models, fontsize=9, rotation=15, ha="right")
    ax1.grid(axis="y", linestyle="--", alpha=0.4)
    for bar in b1:
        h = bar.get_height()
        ax1.annotate(f"{h:.1f}s", xy=(bar.get_x() + bar.get_width()/2, h), xytext=(0, 3),
                     textcoords="offset points", ha="center", va="bottom", fontsize=9, fontweight="bold")

    # Inference Latency
    b2 = ax2.bar(x, infer_latencies, color=colors, width=0.55, edgecolor="black", linewidth=0.6)
    ax2.set_ylabel("Inference Latency (ms / sample)", fontsize=10, fontweight="bold")
    ax2.set_title("Inference Latency (Per Audio Clip)", fontsize=11, fontweight="bold")
    ax2.set_xticks(x)
    ax2.set_xticklabels(models, fontsize=9, rotation=15, ha="right")
    ax2.grid(axis="y", linestyle="--", alpha=0.4)
    for bar in b2:
        h = bar.get_height()
        ax2.annotate(f"{h:.1f}ms", xy=(bar.get_x() + bar.get_width()/2, h), xytext=(0, 3),
                     textcoords="offset points", ha="center", va="bottom", fontsize=9, fontweight="bold")

    fig.tight_layout()
    return fig


def plot_audio_analysis(y: np.ndarray, sr: int = config.SAMPLE_RATE) -> plt.Figure:
    """Plot waveform and mel-spectrogram for uploaded audio."""
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(7.5, 4.0), dpi=150, sharex=True)

    # Waveform
    time_axis = np.linspace(0, len(y) / sr, len(y))
    ax1.plot(time_axis, y, color="#1f77b4", linewidth=0.8)
    ax1.set_title("Normalized Audio Waveform (Time Domain)", fontsize=10, fontweight="bold")
    ax1.set_ylabel("Amplitude", fontsize=9)
    ax1.grid(True, linestyle=":", alpha=0.5)

    # Spectrogram
    melspec = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=64, fmax=sr // 2)
    melspec_db = librosa.power_to_db(melspec, ref=np.max)
    img = librosa.display.specshow(melspec_db, sr=sr, x_axis="time", y_axis="mel", fmax=sr // 2, ax=ax2, cmap="magma")
    ax2.set_title("Mel-Spectrogram (Frequency Domain)", fontsize=10, fontweight="bold")
    ax2.set_ylabel("Frequency (Hz)", fontsize=9)
    ax2.set_xlabel("Time (seconds)", fontsize=9)
    fig.colorbar(img, ax=ax2, format="%+2.0f dB", fraction=0.046, pad=0.04)

    fig.tight_layout()
    return fig
