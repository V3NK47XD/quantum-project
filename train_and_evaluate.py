"""Master training and evaluation orchestrator across CPU, GPU, and Quantum VQC."""

import os
import time
import json
from pathlib import Path
from typing import Dict, Any, List

import numpy as np
import pandas as pd

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
import config
from src.dataset import load_dataset_features, prepare_dataset_pipeline, PREPROCESSORS_PATH
from src.classical_model import (
    TORCH_AVAILABLE,
    get_available_devices,
    train_mlp_model,
    predict_batch,
)
from src.quantum_model import QuantumVoiceClassifier
from src.evaluate import compute_classification_metrics, save_benchmark_results
from src.visualization import (
    plot_confusion_matrix,
    plot_metrics_comparison,
    plot_latency_and_training_comparison,
)


def run_training_and_evaluation() -> Dict[str, Any]:
    print("=" * 60)
    print("  QUANTUM & CLASSICAL ML VOICE ANTI-SPOOFING BENCHMARK")
    print("=" * 60)

    # 1. Load dataset features
    print("\n--- 1. Loading Features with Caching ---")
    X, y, meta = load_dataset_features(use_cache=True)
    print(f"Total samples: {len(X)} (Bonafide: {np.sum(y == 0)}, Spoof: {np.sum(y == 1)})")

    # 2. Split and Preprocess (Zero Data Leakage)
    print("\n--- 2. Dataset Splitting & Preprocessing Pipeline ---")
    data = prepare_dataset_pipeline(
        X, y,
        test_size=config.TEST_SIZE,
        random_state=config.RANDOM_STATE,
        n_pca_components=config.PCA_COMPONENTS,
    )

    X_train_scaled = data["X_train_scaled"]
    X_test_scaled = data["X_test_scaled"]
    X_train_quantum = data["X_train_quantum"]
    X_test_quantum = data["X_test_quantum"]
    y_train = data["y_train"]
    y_test = data["y_test"]

    print(f"Training set: {len(y_train)} samples | Test set: {len(y_test)} samples")
    print(f"Classical feature shape: {X_train_scaled.shape}")
    print(f"Quantum feature shape (PCA-4): {X_train_quantum.shape}")

    benchmark_records = []
    test_predictions_dict = {"true_label": y_test.tolist()}

    # 3. Train Classical MLP on CPU
    print("\n--- 3. Classical MLP (CPU) ---")
    if TORCH_AVAILABLE:
        cpu_save_path = config.MODELS_DIR / "mlp_cpu.pt"
        cpu_res = train_mlp_model(
            X_train=X_train_scaled,
            y_train=y_train,
            X_val=X_test_scaled,
            y_val=y_test,
            device_name="cpu",
            epochs=config.CLASSICAL_EPOCHS,
            batch_size=config.CLASSICAL_BATCH_SIZE,
            save_path=cpu_save_path,
        )

        cpu_preds, cpu_probs, cpu_batch_time = predict_batch(
            cpu_res["model"], X_test_scaled, device_name="cpu"
        )
        test_predictions_dict["pred_mlp_cpu"] = cpu_preds.tolist()

        cpu_metrics = compute_classification_metrics(
            y_true=y_test,
            y_pred=cpu_preds,
            y_prob=cpu_probs,
            training_time_sec=cpu_res["training_time_sec"],
            inference_latency_ms=cpu_res["inference_latency_ms"],
            model_name="PyTorch MLP (CPU)",
        )
        benchmark_records.append(cpu_metrics)
        print(f"CPU MLP: Accuracy = {cpu_metrics['accuracy']:.4f}, F1 = {cpu_metrics['f1_score']:.4f}, Train Time = {cpu_metrics['training_time_sec']:.2f}s, Latency = {cpu_metrics['inference_latency_ms']:.2f}ms")
    else:
        print("[WARNING] PyTorch not installed yet; skipping CPU MLP.")

    # 4. Train Classical MLP on NVIDIA CUDA GPU
    print("\n--- 4. Classical MLP (NVIDIA CUDA GPU) ---")
    devices = get_available_devices()
    if TORCH_AVAILABLE and devices["cuda"]:
        gpu_save_path = config.MODELS_DIR / "mlp_gpu.pt"
        gpu_res = train_mlp_model(
            X_train=X_train_scaled,
            y_train=y_train,
            X_val=X_test_scaled,
            y_val=y_test,
            device_name="cuda",
            epochs=config.CLASSICAL_EPOCHS,
            batch_size=config.CLASSICAL_BATCH_SIZE,
            save_path=gpu_save_path,
        )

        gpu_preds, gpu_probs, gpu_batch_time = predict_batch(
            gpu_res["model"], X_test_scaled, device_name="cuda"
        )
        test_predictions_dict["pred_mlp_gpu"] = gpu_preds.tolist()

        gpu_metrics = compute_classification_metrics(
            y_true=y_test,
            y_pred=gpu_preds,
            y_prob=gpu_probs,
            training_time_sec=gpu_res["training_time_sec"],
            inference_latency_ms=gpu_res["inference_latency_ms"],
            model_name=f"PyTorch MLP (GPU: {devices['gpu_name'] or 'CUDA'})",
        )
        benchmark_records.append(gpu_metrics)
        print(f"GPU MLP: Accuracy = {gpu_metrics['accuracy']:.4f}, F1 = {gpu_metrics['f1_score']:.4f}, Train Time = {gpu_metrics['training_time_sec']:.2f}s, Latency = {gpu_metrics['inference_latency_ms']:.2f}ms")
    else:
        print(f"[INFO] CUDA GPU acceleration not available or PyTorch missing (cuda_available={devices.get('cuda')}).")

    # 5. Train Quantum VQC on Qiskit Aer
    print("\n--- 5. Quantum ML: Variational Quantum Classifier (VQC) ---")
    vqc_save_path = config.MODELS_DIR / "vqc_weights.npy"
    q_classifier = QuantumVoiceClassifier(
        num_qubits=config.N_QUBITS,
        feature_map_type=config.FEATURE_MAP_TYPE,
        ansatz_type="RealAmplitudes",
        ansatz_reps=config.ANSATZ_REPS,
        maxiter=config.QUANTUM_MAX_ITER,
        shots=config.QUANTUM_SHOTS,
        seed=config.QUANTUM_SEED,
    )

    q_classifier.fit(X_train_quantum, y_train)
    q_classifier.save_weights(vqc_save_path)

    # Evaluate Quantum model on held-out test set
    t0_q_infer = time.perf_counter()
    q_preds = q_classifier.predict(X_test_quantum)
    q_probs = q_classifier.predict_proba(X_test_quantum)
    q_infer_time = time.perf_counter() - t0_q_infer
    q_avg_latency_ms = (q_infer_time / len(X_test_quantum)) * 1000.0

    test_predictions_dict["pred_quantum_vqc"] = q_preds.tolist()

    quantum_metrics = compute_classification_metrics(
        y_true=y_test,
        y_pred=q_preds,
        y_prob=q_probs,
        training_time_sec=q_classifier.training_time_sec,
        inference_latency_ms=q_avg_latency_ms,
        model_name="Qiskit VQC (Aer Simulator)",
    )
    benchmark_records.append(quantum_metrics)
    print(f"Quantum VQC: Accuracy = {quantum_metrics['accuracy']:.4f}, F1 = {quantum_metrics['f1_score']:.4f}, Train Time = {quantum_metrics['training_time_sec']:.2f}s, Latency = {quantum_metrics['inference_latency_ms']:.2f}ms")

    # 6. Save Benchmark Results & Prediction CSV
    print("\n--- 6. Saving Evaluation Artifacts ---")
    save_benchmark_results(benchmark_records)

    pred_df = pd.DataFrame(test_predictions_dict)
    pred_csv_path = config.RESULTS_DIR / "test_predictions.csv"
    pred_df.to_csv(pred_csv_path, index=False)
    print(f"Saved test predictions to {pred_csv_path}")

    # Generate and save comparison plots
    fig_metrics = plot_metrics_comparison(benchmark_records)
    fig_metrics.savefig(config.RESULTS_DIR / "metrics_comparison.png", dpi=150)

    fig_latency = plot_latency_and_training_comparison(benchmark_records)
    fig_latency.savefig(config.RESULTS_DIR / "latency_comparison.png", dpi=150)

    for rec in benchmark_records:
        safe_name = rec["model_name"].replace(" ", "_").replace(":", "_").replace("(", "").replace(")", "").lower()
        fig_cm = plot_confusion_matrix(rec["confusion_matrix"], model_name=rec["model_name"])
        fig_cm.savefig(config.RESULTS_DIR / f"cm_{safe_name}.png", dpi=150)

    print("All evaluation plots generated and saved successfully!")
    return {
        "benchmark_records": benchmark_records,
        "test_size": len(y_test),
        "train_size": len(y_train),
    }


if __name__ == "__main__":
    run_training_and_evaluation()
