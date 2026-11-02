"""End-to-end system smoke tests covering audio processing, all 3 models, and artifacts."""

import os
import json
from pathlib import Path
import numpy as np
import torch

import config
from src.features import load_audio, compute_features, extract_features_with_cache
from src.dataset import preprocess_single_sample, PREPROCESSORS_PATH
from src.classical_model import VoiceSpoofMLP, predict_single_sample, get_available_devices
from src.quantum_model import QuantumVoiceClassifier, build_quantum_circuit
from src.evaluate import load_benchmark_results
from src.visualization import (
    plot_audio_analysis,
    plot_confusion_matrix,
    plot_metrics_comparison,
    plot_latency_and_training_comparison,
)


def test_audio_pipeline():
    print("Testing Audio & Feature Pipeline...")
    demo_files = list(config.DEMO_DIR.glob("*.flac"))
    assert len(demo_files) > 0, "No demo audio files found in DEMO_DIR"
    test_audio = demo_files[0]

    # Test load
    y, sr = load_audio(test_audio, sr=config.SAMPLE_RATE, duration=config.DURATION)
    assert len(y) == int(config.SAMPLE_RATE * config.DURATION), f"Audio length mismatch: {len(y)}"
    print(f" Audio loaded successfully: {len(y)} samples @ {sr} Hz")

    # Test feature extraction
    feat_raw, comp = compute_features(y, sr)
    assert feat_raw.shape == (config.TOTAL_FEATURE_DIM,), f"Feature dimension mismatch: {feat_raw.shape}"
    print(f" Computed 70-dimensional features: {feat_raw.shape}")

    # Test caching
    feat_cached, meta = extract_features_with_cache(test_audio, use_cache=True)
    assert meta["cached"] is True, "Feature caching failed"
    print(f" Feature caching verified: {meta['hash'][:12]}...")

    # Test Preprocessing (Zero Leakage)
    assert PREPROCESSORS_PATH.exists(), "Preprocessors file missing"
    scaled_class, scaled_quant = preprocess_single_sample(feat_raw)
    assert scaled_class.shape == (1, 70), f"Scaled classical shape mismatch: {scaled_class.shape}"
    assert scaled_quant.shape == (1, 4), f"Scaled quantum shape mismatch: {scaled_quant.shape}"
    assert np.all(scaled_quant >= 0.0) and np.all(scaled_quant <= np.pi + 1e-4), "Quantum angles outside [0, pi]"
    print(f" Zero-leakage Preprocessor verified: Classical={scaled_class.shape}, Quantum={scaled_quant.shape}")


def test_models():
    print("\nTesting All Three Models (CPU, GPU, Quantum VQC)...")
    demo_audio = list(config.DEMO_DIR.glob("*.flac"))[0]
    feat_raw, _ = extract_features_with_cache(demo_audio)
    scaled_class, scaled_quant = preprocess_single_sample(feat_raw)

    # 1. Classical CPU Model
    cpu_path = config.MODELS_DIR / "mlp_cpu.pt"
    assert cpu_path.exists(), "CPU model checkpoint missing"
    m_cpu = VoiceSpoofMLP(input_dim=config.TOTAL_FEATURE_DIM)
    m_cpu.load_state_dict(torch.load(cpu_path, map_location="cpu", weights_only=True))
    res_cpu = predict_single_sample(m_cpu, scaled_class, device_name="cpu")
    assert res_cpu["predicted_label"] in [0, 1]
    assert 0.0 <= res_cpu["confidence"] <= 1.0
    print(f" CPU MLP: Class='{res_cpu['class_name']}', Conf={res_cpu['confidence']:.3f}, Latency={res_cpu['latency_ms']:.2f}ms")

    # 2. Classical CUDA GPU Model
    devices = get_available_devices()
    print(f" Hardware Detection: CUDA={devices['cuda']}, Device='{devices['gpu_name']}'")
    gpu_path = config.MODELS_DIR / "mlp_gpu.pt"
    if devices["cuda"] and gpu_path.exists():
        m_gpu = VoiceSpoofMLP(input_dim=config.TOTAL_FEATURE_DIM)
        m_gpu.load_state_dict(torch.load(gpu_path, map_location="cuda", weights_only=True))
        res_gpu = predict_single_sample(m_gpu, scaled_class, device_name="cuda")
        assert res_gpu["predicted_label"] in [0, 1]
        assert 0.0 <= res_gpu["confidence"] <= 1.0
        print(f" GPU MLP: Class='{res_gpu['class_name']}', Conf={res_gpu['confidence']:.3f}, Latency={res_gpu['latency_ms']:.2f}ms")

    # 3. Quantum VQC Model
    vqc_path = config.MODELS_DIR / "vqc_weights.npy"
    assert vqc_path.exists(), "VQC weights missing"
    q_clf = QuantumVoiceClassifier(
        num_qubits=config.N_QUBITS,
        feature_map_type=config.FEATURE_MAP_TYPE,
        ansatz_type="RealAmplitudes",
        ansatz_reps=config.ANSATZ_REPS,
    )
    q_clf.load_weights(vqc_path)
    res_q = q_clf.predict_single(scaled_quant)
    assert res_q["predicted_label"] in [0, 1]
    assert 0.0 <= res_q["confidence"] <= 1.0
    print(f" Quantum VQC: Class='{res_q['class_name']}', Conf={res_q['confidence']:.3f}, Latency={res_q['latency_ms']:.2f}ms")


def test_visualizations_and_artifacts():
    print("\nTesting Visualization & Benchmark Artifacts...")
    benchmarks = load_benchmark_results()
    assert benchmarks is not None and len(benchmarks) >= 2, "Benchmark metrics missing or incomplete"
    print(f" Loaded {len(benchmarks)} benchmark entries")

    # Plot metrics
    fig_m = plot_metrics_comparison(benchmarks)
    assert fig_m is not None
    fig_l = plot_latency_and_training_comparison(benchmarks)
    assert fig_l is not None
    fig_cm = plot_confusion_matrix(benchmarks[0]["confusion_matrix"])
    assert fig_cm is not None

    # Quantum circuit drawing
    _, _, full_qc = build_quantum_circuit(num_qubits=4, feature_map_type="ZZFeatureMap", ansatz_type="RealAmplitudes", ansatz_reps=2)
    text_draw = full_qc.draw(output="text")
    assert len(str(text_draw)) > 50, "Quantum circuit text draw empty"
    fig_qc = full_qc.draw(output="mpl")
    assert fig_qc is not None, "Quantum circuit mpl draw failed"
    print(" Quantum circuit visualizer verified (both Text and Matplotlib)")

    # Check export files
    assert (config.RESULTS_DIR / "evaluation_metrics.json").exists()
    assert (config.RESULTS_DIR / "evaluation_metrics.csv").exists()
    assert (config.RESULTS_DIR / "test_predictions.csv").exists()
    print(" All CSV and JSON export files verified")


if __name__ == "__main__":
    test_audio_pipeline()
    test_models()
    test_visualizations_and_artifacts()
    print("\n" + "=" * 50)
    print(" ALL END-TO-END SMOKE TESTS PASSED SUCCESSFULLY! ")
    print("=" * 50)
