"""Streamlit Web Application: Quantum & Classical ML Voice Anti-Spoofing Classifier.

Comprehensive forensic evaluation across:
1. Classical PyTorch Deep Learning (CPU vs NVIDIA CUDA)
2. Variational Quantum Classifier (VQC on Qiskit Aer)
3. Multi-Dataset Corpus: ASVspoof 2019 LA, Kaggle Human/Non-human Voices, CommonVoice Indian Accent
"""

import os
import json
import time
from pathlib import Path
from typing import Dict, Any, Optional, List

import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
import config
from src.features import (
    load_audio,
    compute_features,
    extract_features_with_cache,
)
from src.dataset import (
    preprocess_single_sample,
    PREPROCESSORS_PATH,
)
from src.classical_model import (
    TORCH_AVAILABLE,
    VoiceSpoofMLP,
    get_available_devices,
    predict_single_sample,
)
from src.quantum_model import (
    QuantumVoiceClassifier,
    build_quantum_circuit,
)
from src.evaluate import load_benchmark_results
from src.visualization import (
    plot_audio_analysis,
    plot_confusion_matrix,
    plot_metrics_comparison,
    plot_latency_and_training_comparison,
)

if TORCH_AVAILABLE:
    import torch

# Set Streamlit Page Config
st.set_page_config(
    page_title="Quantum Audio Forensics: AI vs Human Voice Classifier",
    page_icon="🎙",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom Clean CSS Styling
st.markdown("""
<style>
    /* Clean layout styling */
    .block-container {
        padding-top: 1.8rem;
        padding-bottom: 2.5rem;
    }
    .badge-bonafide {
        display: inline-block;
        padding: 0.25rem 0.6rem;
        background-color: rgba(16, 185, 129, 0.15);
        color: #059669;
        border: 1px solid rgba(16, 185, 129, 0.3);
        border-radius: 6px;
        font-weight: 600;
        font-size: 0.85rem;
    }
    .badge-spoof {
        display: inline-block;
        padding: 0.25rem 0.6rem;
        background-color: rgba(239, 68, 68, 0.15);
        color: #dc2626;
        border: 1px solid rgba(239, 68, 68, 0.3);
        border-radius: 6px;
        font-weight: 600;
        font-size: 0.85rem;
    }
    .dataset-card {
        border: 1px solid rgba(128, 128, 128, 0.2);
        border-radius: 8px;
        padding: 1rem;
        height: 100%;
        background-color: rgba(128, 128, 128, 0.03);
    }
    .metric-subtext {
        font-size: 0.85rem;
        color: rgba(128, 128, 128, 0.85);
    }
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def load_system_models():
    """Load pre-trained classical and quantum models."""
    models = {
        "mlp_cpu": None,
        "mlp_gpu": None,
        "quantum_vqc": None,
        "devices": get_available_devices(),
    }

    # Load Classical CPU Model
    cpu_path = config.MODELS_DIR / "mlp_cpu.pt"
    if TORCH_AVAILABLE and cpu_path.exists():
        try:
            m_cpu = VoiceSpoofMLP(input_dim=config.TOTAL_FEATURE_DIM)
            m_cpu.load_state_dict(torch.load(cpu_path, map_location="cpu", weights_only=True))
            m_cpu.eval()
            models["mlp_cpu"] = m_cpu
        except Exception as e:
            st.sidebar.warning(f"Failed to load CPU model: {e}")

    # Load Classical GPU Model
    gpu_path = config.MODELS_DIR / "mlp_gpu.pt"
    if TORCH_AVAILABLE and models["devices"]["cuda"] and gpu_path.exists():
        try:
            m_gpu = VoiceSpoofMLP(input_dim=config.TOTAL_FEATURE_DIM)
            m_gpu.load_state_dict(torch.load(gpu_path, map_location="cuda", weights_only=True))
            m_gpu.eval()
            models["mlp_gpu"] = m_gpu
        except Exception as e:
            st.sidebar.warning(f"Failed to load GPU model: {e}")

    # Load Quantum VQC Model
    vqc_path = config.MODELS_DIR / "vqc_weights.npy"
    if vqc_path.exists():
        try:
            q_clf = QuantumVoiceClassifier(
                num_qubits=config.N_QUBITS,
                feature_map_type=config.FEATURE_MAP_TYPE,
                ansatz_type="RealAmplitudes",
                ansatz_reps=config.ANSATZ_REPS,
            )
            q_clf.load_weights(vqc_path)
            models["quantum_vqc"] = q_clf
        except Exception as e:
            st.sidebar.warning(f"Failed to load Quantum model: {e}")

    return models


def load_dataset_metadata() -> pd.DataFrame:
    """Safely load multi-dataset metadata manifest with fallback."""
    meta_path = config.DATA_DIR / "metadata.json"
    if not meta_path.exists():
        return pd.DataFrame()

    try:
        with open(meta_path, "r", encoding="utf-8") as f:
            records = json.load(f)
        df = pd.DataFrame(records)

        # Standardize source labels for clean UI display
        source_mapping = {
            "asvspoof_2019": "ASVspoof 2019 LA",
            "kaggle_human_voices": "Kaggle Human Voices",
            "kaggle_nonhuman_voices": "Kaggle Non-human Voices",
            "kaggle_human_nonhuman": "Kaggle Voices",
            "commonvoice_indian_accent": "CommonVoice Indian Accent",
        }
        if "dataset_source" in df.columns:
            df["source_display"] = df["dataset_source"].map(lambda x: source_mapping.get(x, x))
        else:
            df["source_display"] = "ASVspoof 2019 LA"

        if "label_name" not in df.columns and "label" in df.columns:
            df["label_name"] = df["label"].map({
                config.LABEL_BONAFIDE: "Human (Bonafide)",
                config.LABEL_SPOOF: "AI-Generated (Spoof)",
            })

        return df
    except Exception as ex:
        st.error(f"Error loading metadata manifest: {ex}")
        return pd.DataFrame()


def main():
    # Top Header
    st.title("Quantum Audio Forensics: AI vs Human Voice Classifier")
    st.markdown(
        "**Comparative Analysis of Classical PyTorch Deep Learning (CPU vs NVIDIA CUDA) "
        "and Variational Quantum Classifiers (VQC on Qiskit Aer) across Multi-Dataset Speech Corpora.**"
    )

    models_bundle = load_system_models()
    devices = models_bundle["devices"]
    df_meta_all = load_dataset_metadata()

    # Sidebar: System Hardware & Model Checkpoints
    with st.sidebar:
        with st.container(border=True):
            st.subheader("System Hardware")
            st.metric(
                label="NVIDIA CUDA Acceleration",
                value="Available" if devices["cuda"] else "Not Detected",
                delta=devices["gpu_name"] if devices["cuda"] else "CPU Only",
            )
            st.metric(
                label="Quantum Simulator",
                value="Qiskit Aer",
                delta=f"{config.N_QUBITS} Qubits (PCA-4)",
            )

        with st.container(border=True):
            st.subheader("Model Checkpoints")
            cpu_status = "[Ready]" if models_bundle["mlp_cpu"] else "[Needs Training]"
            gpu_status = "[Ready]" if models_bundle["mlp_gpu"] else ("[CPU Fallback]" if not devices["cuda"] else "[Needs Training]")
            q_status = "[Ready]" if models_bundle["quantum_vqc"] else "[Needs Training]"

            st.markdown(f"""
            - **PyTorch CPU MLP:** `{cpu_status}`
            - **PyTorch GPU MLP:** `{gpu_status}`
            - **Qiskit VQC:** `{q_status}`
            """)

        with st.container(border=True):
            st.subheader("Corpus Configuration")
            if not df_meta_all.empty:
                n_total = len(df_meta_all)
                n_bonafide = (df_meta_all["label"] == config.LABEL_BONAFIDE).sum() if "label" in df_meta_all.columns else 0
                n_spoof = (df_meta_all["label"] == config.LABEL_SPOOF).sum() if "label" in df_meta_all.columns else 0
                st.write(f"- Total Samples: **{n_total}**")
                st.write(f"- Bonafide Human: **{n_bonafide}**")
                st.write(f"- AI Spoof: **{n_spoof}**")
                st.write(f"- Datasets: **3 Active Sources**")
            else:
                st.caption("Metadata manifest not loaded.")

    # Main Navigation Tabs
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "Audio Classification & Inference",
        "Quantum Circuit Architecture",
        "Evaluation & Benchmarks",
        "Multi-Dataset Corpus & Features",
        "Results Export & Methodology",
    ])

    # ----------------------------------------------------
    # TAB 1: AUDIO CLASSIFICATION & INFERENCE
    # ----------------------------------------------------
    with tab1:
        st.subheader("Audio Input & Signal Analysis")

        with st.container(border=True):
            input_mode = st.radio(
                "Select Voice Input Source:",
                [
                    "Live Microphone Recording",
                    "Upload Voice Audio File",
                    "Benchmark Demo Samples",
                ],
                horizontal=True,
            )

            audio_bytes = None
            source_label = "Custom Audio"

            if input_mode == "Live Microphone Recording":
                st.markdown("**Live Microphone Recording**")
                st.caption("Click the microphone button to record audio directly in the browser, then stop to submit.")
                mic_audio = st.audio_input("Record Audio:")
                if mic_audio is not None:
                    audio_bytes = mic_audio.read()
                    source_label = "Live Microphone Recording"

            elif input_mode == "Upload Voice Audio File":
                st.markdown("**Voice Audio File Uploader**")
                uploaded_file = st.file_uploader(
                    "Upload voice clip (.flac, .wav, .mp3, .ogg):",
                    type=["flac", "wav", "mp3", "ogg"],
                )
                if uploaded_file is not None:
                    audio_bytes = uploaded_file.read()
                    source_label = f"Uploaded File: {uploaded_file.name}"

            else:
                st.markdown("**Multi-Dataset Benchmark Demo Samples**")
                demo_files = sorted([f for ext in ("*.flac", "*.mp3", "*.wav") for f in config.DEMO_DIR.glob(ext)])

                # Friendly display labels mapping
                demo_label_map = {
                    "demo_human_1.flac": "ASVspoof 2019 - Bonafide Human Sample 1 (FLAC)",
                    "demo_human_2.flac": "ASVspoof 2019 - Bonafide Human Sample 2 (FLAC)",
                    "demo_ai_spoof_1.flac": "ASVspoof 2019 - AI Synthesized Spoof Sample 1 (FLAC)",
                    "demo_ai_spoof_2.flac": "ASVspoof 2019 - AI Synthesized Spoof Sample 2 (FLAC)",
                    "demo_human_indian.mp3": "CommonVoice Indian Accent - Bonafide Human (MP3)",
                    "demo_ai_kaggle.mp3": "Kaggle Non-human Voices - AI Spoof (MP3)",
                }

                demo_names = [f.name for f in demo_files]
                selected_demo_display = st.selectbox(
                    "Choose benchmark sample across datasets:",
                    options=demo_names,
                    format_func=lambda x: demo_label_map.get(x, x),
                )
                if selected_demo_display:
                    chosen_path = config.DEMO_DIR / selected_demo_display
                    with open(chosen_path, "rb") as f:
                        audio_bytes = f.read()
                    source_label = demo_label_map.get(selected_demo_display, selected_demo_display)

        if audio_bytes is not None:
            with st.container(border=True):
                st.markdown(f"**Loaded Audio Source:** `{source_label}`")
                st.audio(audio_bytes)

            # Signal waveform & spectrogram
            try:
                y, sr = load_audio(audio_bytes, sr=config.SAMPLE_RATE, duration=config.DURATION)

                with st.container(border=True):
                    st.markdown("##### Audio Waveform & Mel-Spectrogram (Librosa)")
                    fig_audio = plot_audio_analysis(y, sr)
                    st.pyplot(fig_audio)
                    plt.close(fig_audio)

                # Feature extraction
                feat_raw, meta = extract_features_with_cache(audio_bytes, use_cache=True)
                components = meta["components"]

                with st.expander("Acoustic Features Breakdown (70 Dimensions)", expanded=False):
                    c1, c2, c3 = st.columns(3)
                    with c1:
                        with st.container(border=True):
                            st.markdown("**MFCC (20 Coefficients)**")
                            st.dataframe(pd.DataFrame({
                                "Mean": components["mfcc_mean"].round(3),
                                "Std": components["mfcc_std"].round(3),
                            }), height=220)
                    with c2:
                        with st.container(border=True):
                            st.markdown("**Spectral Properties**")
                            st.write(f"- Centroid Mean: `{components['centroid_mean'][0]:.2f} Hz`")
                            st.write(f"- Centroid Std: `{components['centroid_std'][0]:.2f} Hz`")
                            st.write(f"- Bandwidth Mean: `{components['bandwidth_mean'][0]:.2f} Hz`")
                            st.write(f"- Bandwidth Std: `{components['bandwidth_std'][0]:.2f} Hz`")
                            st.write(f"- Zero Crossing Rate: `{components['zcr_mean'][0]:.4f}`")
                    with c3:
                        with st.container(border=True):
                            st.markdown("**Chroma STFT (12 Pitch Classes)**")
                            st.dataframe(pd.DataFrame({
                                "Chroma Mean": components["chroma_mean"].round(3),
                                "Chroma Std": components["chroma_std"].round(3),
                            }), height=220)

                # Classification Section
                st.subheader("Triple Model Classification (CPU vs GPU vs Quantum)")

                if not PREPROCESSORS_PATH.exists():
                    st.error("Preprocessing pipeline not found. Please train models first via `python train_and_evaluate.py`.")
                else:
                    feat_scaled, feat_quantum = preprocess_single_sample(feat_raw)

                    col_m1, col_m2, col_m3 = st.columns(3)
                    pred_cpu = None
                    pred_gpu = None
                    pred_q = None

                    # Model 1: Classical CPU MLP
                    with col_m1:
                        with st.container(border=True):
                            st.markdown("#### Classical MLP (CPU)")
                            if models_bundle["mlp_cpu"] is not None:
                                pred_cpu = predict_single_sample(models_bundle["mlp_cpu"], feat_scaled, device_name="cpu")
                                is_spoof = pred_cpu["predicted_label"] == config.LABEL_SPOOF
                                badge_html = (
                                    '<span class="badge-spoof">AI-Generated (Spoof)</span>'
                                    if is_spoof
                                    else '<span class="badge-bonafide">Human (Bonafide)</span>'
                                )
                                st.markdown(f"**Predicted Origin:** {badge_html}", unsafe_allow_html=True)
                                st.metric(
                                    label="Confidence",
                                    value=f"{pred_cpu['confidence']*100:.1f}%",
                                    delta=f"{pred_cpu['latency_ms']:.2f} ms latency",
                                )
                                st.progress(float(pred_cpu["confidence"]))
                                st.caption(
                                    f"Probabilities: Human: {pred_cpu['probabilities'][config.LABEL_NAMES[0]]:.3f} | "
                                    f"AI: {pred_cpu['probabilities'][config.LABEL_NAMES[1]]:.3f}"
                                )
                            else:
                                st.info("CPU MLP weights not found. Train models first.")

                    # Model 2: Classical GPU MLP
                    with col_m2:
                        with st.container(border=True):
                            st.markdown("#### Classical MLP (CUDA GPU)")
                            if models_bundle["mlp_gpu"] is not None:
                                pred_gpu = predict_single_sample(models_bundle["mlp_gpu"], feat_scaled, device_name="cuda")
                                is_spoof = pred_gpu["predicted_label"] == config.LABEL_SPOOF
                                badge_html = (
                                    '<span class="badge-spoof">AI-Generated (Spoof)</span>'
                                    if is_spoof
                                    else '<span class="badge-bonafide">Human (Bonafide)</span>'
                                )
                                st.markdown(f"**Predicted Origin:** {badge_html}", unsafe_allow_html=True)
                                st.metric(
                                    label="Confidence",
                                    value=f"{pred_gpu['confidence']*100:.1f}%",
                                    delta=f"{pred_gpu['latency_ms']:.2f} ms latency",
                                )
                                st.progress(float(pred_gpu["confidence"]))
                                st.caption(
                                    f"Probabilities: Human: {pred_gpu['probabilities'][config.LABEL_NAMES[0]]:.3f} | "
                                    f"AI: {pred_gpu['probabilities'][config.LABEL_NAMES[1]]:.3f}"
                                )
                            elif models_bundle["mlp_cpu"] is not None and not devices["cuda"]:
                                st.caption("NVIDIA CUDA GPU not detected on this system. Showing CPU benchmark.")
                                is_spoof = pred_cpu["predicted_label"] == config.LABEL_SPOOF if pred_cpu else False
                                badge_html = (
                                    '<span class="badge-spoof">AI-Generated (Spoof)</span>'
                                    if is_spoof
                                    else '<span class="badge-bonafide">Human (Bonafide)</span>'
                                )
                                st.markdown(f"**Predicted Origin (CPU):** {badge_html}", unsafe_allow_html=True)
                                if pred_cpu:
                                    st.metric(
                                        label="Confidence (CPU)",
                                        value=f"{pred_cpu['confidence']*100:.1f}%",
                                        delta=f"{pred_cpu['latency_ms']:.2f} ms",
                                    )
                                    st.progress(float(pred_cpu["confidence"]))
                            else:
                                st.info("GPU MLP weights not found. Train models first.")

                    # Model 3: Quantum VQC
                    with col_m3:
                        with st.container(border=True):
                            st.markdown("#### Quantum VQC (Qiskit Aer)")
                            if models_bundle["quantum_vqc"] is not None:
                                pred_q = models_bundle["quantum_vqc"].predict_single(feat_quantum)
                                is_spoof = pred_q["predicted_label"] == config.LABEL_SPOOF
                                badge_html = (
                                    '<span class="badge-spoof">AI-Generated (Spoof)</span>'
                                    if is_spoof
                                    else '<span class="badge-bonafide">Human (Bonafide)</span>'
                                )
                                st.markdown(f"**Predicted Origin:** {badge_html}", unsafe_allow_html=True)
                                st.metric(
                                    label="Confidence",
                                    value=f"{pred_q['confidence']*100:.1f}%",
                                    delta=f"{pred_q['latency_ms']:.2f} ms latency",
                                )
                                st.progress(float(pred_q["confidence"]))
                                st.caption(
                                    f"Probabilities: Human: {pred_q['probabilities'][config.LABEL_NAMES[0]]:.3f} | "
                                    f"AI: {pred_q['probabilities'][config.LABEL_NAMES[1]]:.3f}"
                                )
                            else:
                                st.info("Quantum VQC weights not found. Train models first.")

                    # Comparative Inference Timing Section
                    with st.container(border=True):
                        st.subheader("Real-Time Inference Latency & Speed Benchmark")

                        timing_rows = []
                        if pred_cpu is not None:
                            timing_rows.append({
                                "Model": "Classical PyTorch MLP (CPU)",
                                "Prediction": "AI-Generated (Spoof)" if pred_cpu["predicted_label"] == config.LABEL_SPOOF else "Human (Bonafide)",
                                "Confidence": f"{pred_cpu['confidence']*100:.1f}%",
                                "Latency (ms)": pred_cpu["latency_ms"],
                                "Hardware / Backend": "Intel/AMD CPU",
                                "Complexity": "6,818 weights",
                            })
                        if pred_gpu is not None:
                            timing_rows.append({
                                "Model": "Classical PyTorch MLP (CUDA GPU)",
                                "Prediction": "AI-Generated (Spoof)" if pred_gpu["predicted_label"] == config.LABEL_SPOOF else "Human (Bonafide)",
                                "Confidence": f"{pred_gpu['confidence']*100:.1f}%",
                                "Latency (ms)": pred_gpu["latency_ms"],
                                "Hardware / Backend": f"NVIDIA CUDA ({devices['gpu_name'] or 'GPU'})",
                                "Complexity": "6,818 weights",
                            })
                        if pred_q is not None:
                            timing_rows.append({
                                "Model": "Quantum VQC (Qiskit Aer)",
                                "Prediction": "AI-Generated (Spoof)" if pred_q["predicted_label"] == config.LABEL_SPOOF else "Human (Bonafide)",
                                "Confidence": f"{pred_q['confidence']*100:.1f}%",
                                "Latency (ms)": pred_q["latency_ms"],
                                "Hardware / Backend": "Qiskit Aer Simulator (4 Qubits)",
                                "Complexity": "12 variational angles",
                            })

                        if timing_rows:
                            lat_cols = st.columns(len(timing_rows))
                            for idx, r in enumerate(timing_rows):
                                with lat_cols[idx]:
                                    st.metric(
                                        label=f"{r['Model']} Latency",
                                        value=f"{r['Latency (ms)']:.2f} ms",
                                        delta=r["Prediction"],
                                    )

                            if pred_cpu is not None and pred_q is not None and pred_cpu["latency_ms"] > 0:
                                speedup = pred_q["latency_ms"] / pred_cpu["latency_ms"]
                                st.info(
                                    f"**Inference Timing Analysis:** Classical MLP executed in **{pred_cpu['latency_ms']:.2f} ms**, "
                                    f"while Quantum VQC executed in **{pred_q['latency_ms']:.2f} ms** "
                                    f"(Classical is **{speedup:.1f}x** faster than quantum statevector simulation on classical CPU; "
                                    f"Quantum VQC operates in Hilbert space H = C^16 using only **12 variational angles**)."
                                )

                            col_chart, col_tbl = st.columns([1, 1])
                            with col_chart:
                                fig_live_lat, ax_ll = plt.subplots(figsize=(6, 3), dpi=130)
                                m_labels = [r["Model"].replace("Classical PyTorch ", "").replace("Qiskit ", "") for r in timing_rows]
                                m_lats = [r["Latency (ms)"] for r in timing_rows]
                                m_colors = ["#2563eb" if "CPU" in r["Model"] else "#059669" if "GPU" in r["Model"] else "#7c3aed" for r in timing_rows]
                                bars = ax_ll.barh(m_labels, m_lats, color=m_colors, height=0.45)
                                ax_ll.set_xlabel("Latency (ms) — lower is faster", fontsize=9)
                                ax_ll.set_title("Single-Sample Inference Latency", fontsize=10, fontweight="bold")
                                ax_ll.grid(axis="x", linestyle="--", alpha=0.4)
                                for b in bars:
                                    w = b.get_width()
                                    ax_ll.text(w + (max(m_lats) * 0.03), b.get_y() + b.get_height()/2, f"{w:.2f} ms", va="center", fontsize=9, fontweight="bold")
                                plt.tight_layout()
                                st.pyplot(fig_live_lat)
                                plt.close(fig_live_lat)

                            with col_tbl:
                                display_df = pd.DataFrame([{
                                    "Model": r["Model"],
                                    "Prediction": r["Prediction"],
                                    "Confidence": r["Confidence"],
                                    "Latency": f"{r['Latency (ms)']:.2f} ms",
                                    "Parameters": r["Complexity"],
                                } for r in timing_rows])
                                st.dataframe(display_df, use_container_width=True, hide_index=True)

            except Exception as ex:
                st.error(f"Error processing audio sample: {ex}")
                import traceback
                st.code(traceback.format_exc())
        else:
            with st.container(border=True):
                st.info("Select a demo sample, record live audio, or upload an audio file above to begin.")

    # ----------------------------------------------------
    # TAB 2: QUANTUM CIRCUIT ARCHITECTURE
    # ----------------------------------------------------
    with tab2:
        st.subheader("Variational Quantum Circuit (VQC) Architecture")

        with st.container(border=True):
            st.markdown("""
            The quantum classifier operates on **4 qubits** initialized in state $|0000\\rangle$.
            1. **Dimensionality Reduction:** 70 acoustic features are projected to 4 principal components via PCA.
            2. **Quantum Feature Map:** Features are mapped into single-qubit rotation angles $\\theta_i \\in [0, \\pi]$ followed by entangling phase gates.
            3. **Parameterized Ansatz:** Trainable $R_y$ rotation layers followed by a CNOT entangling ladder.
            4. **Measurement & Parity Decoding:** Qubit expectation values are sampled on **Qiskit Aer Simulator**. Even parity bitstrings evaluate to Bonafide Human, odd parity to AI Spoof.
            """)

        with st.container(border=True):
            col_q1, col_q2 = st.columns([1, 1])
            with col_q1:
                fm_choice = st.selectbox(
                    "Feature Map Type:",
                    ["ZZFeatureMap", "CustomRotation"],
                    index=0,
                    help="ZZFeatureMap provides non-linear 2nd-order phase entanglement. CustomRotation uses explicit Ry-CNOT-Rz rotations.",
                )
            with col_q2:
                ansatz_reps = st.slider(
                    "Ansatz Repetitions (Circuit Depth):",
                    min_value=1,
                    max_value=4,
                    value=config.ANSATZ_REPS,
                    help="Controls the number of variational rotation and CNOT entanglement layers.",
                )

            fm_qc, ansatz_qc, full_qc = build_quantum_circuit(
                num_qubits=config.N_QUBITS,
                feature_map_type=fm_choice,
                ansatz_type="RealAmplitudes",
                ansatz_reps=ansatz_reps,
            )

            st.markdown(
                f"**Circuit Specifications:** Qubits = `{config.N_QUBITS}` | "
                f"Circuit Depth = `{full_qc.depth()}` | "
                f"Total Quantum Gates = `{full_qc.size()}` | "
                f"Trainable Parameters = `{ansatz_qc.num_parameters}`"
            )

        col_draw1, col_draw2 = st.columns([2, 1])
        with col_draw1:
            with st.container(border=True):
                st.markdown("**Quantum Circuit Diagram (Matplotlib Render)**")
                try:
                    fig_qc = full_qc.draw(output="mpl")
                    st.pyplot(fig_qc)
                    plt.close(fig_qc)
                except Exception as ex:
                    st.warning(f"Matplotlib circuit rendering fallback: {ex}")
                    st.text(str(full_qc.draw(output="text")))

        with col_draw2:
            with st.container(border=True):
                st.markdown("**Quantum Circuit (Text Representation)**")
                st.text(str(full_qc.draw(output="text")))

        # Interactive Quantum Statevector Simulation
        with st.container(border=True):
            st.subheader("Interactive Quantum State & Basis Probability Simulation")
            st.caption("Simulate the statevector across all 16 computational basis states (|0000> to |1111>) on Qiskit Aer.")

            c_sim1, c_sim2, c_sim3, c_sim4 = st.columns(4)
            with c_sim1:
                x0 = st.slider("Qubit 0 Angle (x[0])", 0.0, float(np.pi), 0.85, 0.05)
            with c_sim2:
                x1 = st.slider("Qubit 1 Angle (x[1])", 0.0, float(np.pi), 1.45, 0.05)
            with c_sim3:
                x2 = st.slider("Qubit 2 Angle (x[2])", 0.0, float(np.pi), 2.10, 0.05)
            with c_sim4:
                x3 = st.slider("Qubit 3 Angle (x[3])", 0.0, float(np.pi), 0.65, 0.05)

            weights_path = config.MODELS_DIR / "vqc_weights.npy"
            trained_weights = np.load(weights_path) if weights_path.exists() else np.zeros(ansatz_qc.num_parameters)

            try:
                from qiskit.quantum_info import Statevector

                param_dict = {}
                x_inputs = [x0, x1, x2, x3]
                for p in full_qc.parameters:
                    if "x[" in p.name:
                        idx = int(p.name.split("[")[1].split("]")[0])
                        param_dict[p] = x_inputs[idx]
                    elif "θ[" in p.name:
                        idx = int(p.name.split("[")[1].split("]")[0])
                        param_dict[p] = float(trained_weights[idx]) if idx < len(trained_weights) else 0.0
                    else:
                        param_dict[p] = 0.0

                bound_circuit = full_qc.assign_parameters(param_dict)
                sv = Statevector.from_instruction(bound_circuit)
                probs = sv.probabilities()

                # Parity breakdown
                even_p = sum(p for i, p in enumerate(probs) if format(i, "04b").count("1") % 2 == 0)
                odd_p = 1.0 - even_p

                col_sv_res1, col_sv_res2 = st.columns([1, 1])
                with col_sv_res1:
                    st.markdown("**Parity Classification Outcome:**")
                    st.write(f"- $P(\\text{{Human / Even Parity}}) = \\mathbf{{{even_p*100:.2f}\\%}}$")
                    st.write(f"- $P(\\text{{AI Spoof / Odd Parity}}) = \\mathbf{{{odd_p*100:.2f}\\%}}$")
                    pred_class = "Human (Bonafide)" if even_p >= 0.5 else "AI-Generated (Spoof)"
                    st.info(f"Quantum Decision: **{pred_class}** (Confidence: {max(even_p, odd_p)*100:.1f}%)")

                with col_sv_res2:
                    # Bar chart of basis states
                    fig_sv, ax_sv = plt.subplots(figsize=(7, 3), dpi=130)
                    basis_labels = [format(i, "04b") for i in range(16)]
                    bar_colors = ["#059669" if b.count("1") % 2 == 0 else "#dc2626" for b in basis_labels]
                    ax_sv.bar(basis_labels, probs, color=bar_colors, width=0.6)
                    ax_sv.set_xlabel("Basis State (|q3 q2 q1 q0>)", fontsize=8)
                    ax_sv.set_ylabel("Probability", fontsize=8)
                    ax_sv.set_title("16-Basis State Probability Distribution (Green=Even/Human, Red=Odd/Spoof)", fontsize=9, fontweight="bold")
                    ax_sv.tick_params(axis="x", rotation=45, labelsize=7)
                    ax_sv.grid(axis="y", linestyle="--", alpha=0.4)
                    plt.tight_layout()
                    st.pyplot(fig_sv)
                    plt.close(fig_sv)

            except Exception as ex:
                st.warning(f"Interactive statevector computation note: {ex}")

        with st.container(border=True):
            st.markdown("##### Mathematical Formulation")
            st.latex(r"|\psi(x, \theta)\rangle = U_{\text{Ansatz}}(\theta) \cdot U_{\Phi}(x) |0\rangle^{\otimes 4}")
            st.latex(r"P(y=0|x, \theta) = \sum_{z \in \text{Even Parity}} |\langle z | \psi(x, \theta) \rangle|^2")

    # ----------------------------------------------------
    # TAB 3: EVALUATION & BENCHMARKS
    # ----------------------------------------------------
    with tab3:
        st.subheader("Empirical Performance Comparison on Identical Held-Out Test Set")
        benchmark_results = load_benchmark_results()

        if benchmark_results:
            with st.container(border=True):
                st.markdown("#### Metrics Summary Table")
                summary_rows = []
                for r in benchmark_results:
                    summary_rows.append({
                        "Model": r["model_name"],
                        "Accuracy": f"{r['accuracy']:.4f}",
                        "Precision": f"{r['precision']:.4f}",
                        "Recall": f"{r['recall']:.4f}",
                        "F1-Score": f"{r['f1_score']:.4f}",
                        "Training Time": f"{r['training_time_sec']:.2f} s",
                        "Inference Latency": f"{r['inference_latency_ms']:.2f} ms/sample",
                    })
                st.table(pd.DataFrame(summary_rows))

            col_p1, col_p2 = st.columns(2)
            with col_p1:
                with st.container(border=True):
                    st.markdown("#### Performance Metrics")
                    fig_m = plot_metrics_comparison(benchmark_results)
                    st.pyplot(fig_m)
                    plt.close(fig_m)

            with col_p2:
                with st.container(border=True):
                    st.markdown("#### Training & Inference Latency")
                    fig_l = plot_latency_and_training_comparison(benchmark_results)
                    st.pyplot(fig_l)
                    plt.close(fig_l)

            with st.container(border=True):
                st.markdown("#### Confusion Matrices")
                cm_cols = st.columns(len(benchmark_results))
                for i, r in enumerate(benchmark_results):
                    with cm_cols[i]:
                        fig_cm = plot_confusion_matrix(r["confusion_matrix"], model_name=r["model_name"])
                        st.pyplot(fig_cm)
                        plt.close(fig_cm)
                        st.caption(
                            f"**TN (Human):** {r['true_negatives']} | **FP:** {r['false_positives']}<br>"
                            f"**FN:** {r['false_negatives']} | **TP (Spoof):** {r['true_positives']}",
                            unsafe_allow_html=True,
                        )

            # Cross-Dataset Generalization Breakdown
            pred_file = config.RESULTS_DIR / "test_predictions.csv"
            if pred_file.exists():
                with st.container(border=True):
                    st.subheader("Cross-Dataset Generalization Breakdown")
                    st.caption("Empirical accuracy on the held-out test set evaluated per dataset source.")
                    df_preds = pd.read_csv(pred_file)

                    src_labels = {
                        "asvspoof_2019": "ASVspoof 2019 LA",
                        "kaggle_human_nonhuman": "Kaggle Voices (Human & Non-human)",
                        "commonvoice_indian_accent": "CommonVoice Indian Accent",
                    }

                    cross_rows = []
                    for src in df_preds["dataset_source"].unique():
                        sub = df_preds[df_preds["dataset_source"] == src]
                        n_samples = len(sub)
                        acc_cpu = (sub["true_label"] == sub["pred_mlp_cpu"]).mean() if "pred_mlp_cpu" in sub.columns else np.nan
                        acc_q = (sub["true_label"] == sub["pred_quantum_vqc"]).mean() if "pred_quantum_vqc" in sub.columns else np.nan

                        cross_rows.append({
                            "Dataset Source": src_labels.get(src, src),
                            "Test Samples": n_samples,
                            "Classical MLP (CPU) Accuracy": f"{acc_cpu*100:.1f}%" if not np.isnan(acc_cpu) else "N/A",
                            "Quantum VQC (Aer) Accuracy": f"{acc_q*100:.1f}%" if not np.isnan(acc_q) else "N/A",
                        })

                    st.table(pd.DataFrame(cross_rows))

                # Interactive Test Prediction Explorer
                with st.container(border=True):
                    st.subheader("Interactive Test Prediction Explorer")
                    c_f1, c_f2, c_f3 = st.columns(3)
                    with c_f1:
                        src_filter = st.selectbox(
                            "Filter by Dataset:",
                            options=["All Datasets"] + list(df_preds["dataset_source"].unique()),
                            format_func=lambda x: src_labels.get(x, x),
                        )
                    with c_f2:
                        model_filter = st.selectbox("Inspect Model:", ["Classical MLP (CPU)", "Quantum VQC (Aer)"])
                    with c_f3:
                        outcome_filter = st.selectbox("Prediction Outcome:", ["All Samples", "Correct Only", "Misclassified Only"])

                    df_view = df_preds.copy()
                    if src_filter != "All Datasets":
                        df_view = df_view[df_view["dataset_source"] == src_filter]

                    pred_col = "pred_mlp_cpu" if "CPU" in model_filter else "pred_quantum_vqc"
                    if pred_col in df_view.columns:
                        if outcome_filter == "Correct Only":
                            df_view = df_view[df_view["true_label"] == df_view[pred_col]]
                        elif outcome_filter == "Misclassified Only":
                            df_view = df_view[df_view["true_label"] != df_view[pred_col]]

                    df_view["Ground Truth"] = df_view["true_label"].map({0: "Human (Bonafide)", 1: "AI-Generated (Spoof)"})
                    if pred_col in df_view.columns:
                        df_view["Prediction"] = df_view[pred_col].map({0: "Human (Bonafide)", 1: "AI-Generated (Spoof)"})
                        df_view["Status"] = np.where(df_view["true_label"] == df_view[pred_col], "Correct", "Error")

                    cols_to_show = [c for c in ["filename", "dataset_source", "Ground Truth", "Prediction", "Status"] if c in df_view.columns]
                    st.dataframe(df_view[cols_to_show].head(30), use_container_width=True, hide_index=True)
                    st.caption(f"Showing {min(30, len(df_view))} of {len(df_view)} matching test samples.")

        else:
            with st.container(border=True):
                st.info("No benchmark results found. Run `python train_and_evaluate.py` to generate real empirical metrics.")

    # ----------------------------------------------------
    # TAB 4: MULTI-DATASET CORPUS & FEATURES
    # ----------------------------------------------------
    with tab4:
        st.subheader("Multi-Dataset Audio Corpus & Feature Analysis")
        st.markdown(
            "The system is trained and benchmarked across **three distinct audio corpora** to prevent single-source bias "
            "and rigorously test generalization to synthetic TTS, voice conversion, and diverse human accents."
        )

        # Three Dataset Cards in Containers
        c_d1, c_d2, c_d3 = st.columns(3)
        with c_d1:
            with st.container(border=True):
                st.markdown("#### ASVspoof 2019 LA")
                st.markdown("**Domain:** Classical Anti-Spoofing Benchmark")
                st.write("- **Total Samples:** 200 (100 Bonafide, 100 Spoof)")
                st.write("- **Attacks:** TTS (neural/concatenative) + Voice Conversion")
                st.write("- **Audio Format:** 16 kHz Mono FLAC")
                st.write("- **Provider:** Edinburgh DataShare / Hugging Face")
                st.caption("[Hugging Face Repository](https://huggingface.co/datasets/SpeechAntiSpoofingBenchmarks/ASVspoof2019_LA)")

        with c_d2:
            with st.container(border=True):
                st.markdown("#### Kaggle Voices Corpus")
                st.markdown("**Domain:** Synthetic vs Human Voices")
                st.write("- **Total Samples:** 300 (100 Bonafide, 200 Spoof)")
                st.write("- **Attacks:** Modern AI generated / non-human speech")
                st.write("- **Audio Format:** 16 kHz Mono MP3")
                st.write("- **Provider:** Kaggle (Abdurazzoqov)")
                st.caption("[Kaggle Dataset](https://www.kaggle.com/datasets/aabdurazzoq/human-and-nonhuman-voices)")

        with c_d3:
            with st.container(border=True):
                st.markdown("#### CommonVoice Indian Accent")
                st.markdown("**Domain:** Accent & Demographic Robustness")
                st.write("- **Total Samples:** 100 (100 Bonafide Human)")
                st.write("- **Target:** Cross-accent generalization")
                st.write("- **Audio Format:** 16 kHz Mono MP3")
                st.write("- **Provider:** Mozilla Common Voice / Hugging Face")
                st.caption("[Hugging Face Repository](https://huggingface.co/datasets/ishands/commonvoice-indian_accent)")

        # Overall Corpus Metrics
        meta_path = config.DATA_DIR / "metadata.json"
        if meta_path.exists():
            df_meta = load_dataset_metadata()

            with st.container(border=True):
                d_col1, d_col2, d_col3, d_col4 = st.columns(4)
                d_col1.metric("Total Audio Corpus", len(df_meta))
                d_col2.metric("Bonafide (Human Speech)", int((df_meta["label"] == config.LABEL_BONAFIDE).sum()))
                d_col3.metric("AI-Generated (Spoof)", int((df_meta["label"] == config.LABEL_SPOOF).sum()))
                d_col4.metric("Dataset Sources", len(df_meta["dataset_source"].unique()) if "dataset_source" in df_meta.columns else 1)

            # Class Distribution Plot across Datasets
            with st.container(border=True):
                st.markdown("#### Corpus Class Distribution by Dataset Source")
                cross_tab = pd.crosstab(df_meta["source_display"], df_meta["label_name"])

                fig_dist, ax_dist = plt.subplots(figsize=(7, 2.8), dpi=130)
                cross_tab.plot(kind="bar", stacked=True, ax=ax_dist, color=["#dc2626", "#059669"], width=0.45)
                ax_dist.set_ylabel("Audio Sample Count", fontsize=9, fontweight="bold")
                ax_dist.set_xlabel("Dataset Repository", fontsize=9, fontweight="bold")
                ax_dist.set_title("Multi-Dataset Class Balance (Human vs AI Spoof)", fontsize=10, fontweight="bold")
                ax_dist.tick_params(axis="x", rotation=15, labelsize=8)
                ax_dist.grid(axis="y", linestyle="--", alpha=0.4)
                ax_dist.legend(title="Class", fontsize=8, title_fontsize=8)
                plt.tight_layout()
                st.pyplot(fig_dist)
                plt.close(fig_dist)

            # Interactive Metadata Browser
            with st.container(border=True):
                st.markdown("#### Interactive Corpus Manifest Browser")
                c_mf1, c_mf2, c_mf3 = st.columns(3)
                with c_mf1:
                    src_choices = ["All Datasets"] + list(df_meta["source_display"].unique())
                    chosen_src = st.selectbox("Filter Source:", src_choices)
                with c_mf2:
                    label_choices = ["All Classes"] + list(df_meta["label_name"].unique())
                    chosen_label = st.selectbox("Filter Class:", label_choices)
                with c_mf3:
                    search_term = st.text_input("Search Filename / Utterance ID:", "")

                df_filtered = df_meta.copy()
                if chosen_src != "All Datasets":
                    df_filtered = df_filtered[df_filtered["source_display"] == chosen_src]
                if chosen_label != "All Classes":
                    df_filtered = df_filtered[df_filtered["label_name"] == chosen_label]
                if search_term.strip():
                    df_filtered = df_filtered[
                        df_filtered["filename"].str.contains(search_term, case=False, na=False) |
                        df_filtered["utterance_id"].str.contains(search_term, case=False, na=False)
                    ]

                display_cols = [c for c in ["utterance_id", "filename", "source_display", "label_name"] if c in df_filtered.columns]
                st.dataframe(df_filtered[display_cols].head(50), use_container_width=True, hide_index=True)
                st.caption(f"Displaying {min(50, len(df_filtered))} of {len(df_filtered)} matching audio records.")

                # Audio Sample Player
                if not df_filtered.empty:
                    st.markdown("**Listen to Sample from Corpus:**")
                    sample_options = df_filtered["filename"].tolist()[:25]
                    chosen_sample = st.selectbox("Select file to preview audio:", options=sample_options)
                    if chosen_sample:
                        sample_row = df_filtered[df_filtered["filename"] == chosen_sample].iloc[0]
                        sample_path = config.BASE_DIR / sample_row.get("file_path", "")
                        if sample_path.exists():
                            with open(sample_path, "rb") as f_s:
                                st.audio(f_s.read())
                            st.caption(f"Playing `{chosen_sample}` | Class: `{sample_row.get('label_name')}` | Source: `{sample_row.get('source_display')}`")

        # PCA Feature Analysis
        if PREPROCESSORS_PATH.exists():
            with st.container(border=True):
                import joblib
                pipe = joblib.load(PREPROCESSORS_PATH)
                st.markdown("#### PCA Dimensionality Reduction (70 Acoustic Features to 4 Qubits)")
                var_ratios = pipe.get("explained_variance_ratio", [])
                df_pca = pd.DataFrame({
                    "Principal Component": [f"PC{i+1}" for i in range(len(var_ratios))],
                    "Explained Variance Ratio": [f"{v:.4f}" for v in var_ratios],
                    "Cumulative Variance": [f"{c*100:.2f}%" for c in np.cumsum(var_ratios)],
                })
                st.table(df_pca)
                st.caption(f"Total acoustic variance retained across 4 qubits: **{pipe.get('cumulative_variance', 0)*100:.2f}%**")

    # ----------------------------------------------------
    # TAB 5: RESULTS EXPORT & METHODOLOGY
    # ----------------------------------------------------
    with tab5:
        st.subheader("Export Benchmark Results & Research Methodology")

        # Publications & Reports Download Cards
        col_doc1, col_doc2 = st.columns(2)
        pdf_paper_path = config.BASE_DIR / "Quantum_Voice_Classification_Research_Paper.pdf"
        pdf_report_path = config.BASE_DIR / "Quantum_Voice_Classification_Project_Report.pdf"

        with col_doc1:
            with st.container(border=True):
                st.markdown("#### IEEE / Springer Research Paper (PDF)")
                st.write("Full publication-grade paper with mathematical formulation, circuit diagrams, and benchmark analysis.")
                if pdf_paper_path.exists():
                    with open(pdf_paper_path, "rb") as f_pdf:
                        st.download_button(
                            "Download Research Paper (PDF)",
                            data=f_pdf.read(),
                            file_name="Quantum_Voice_Classification_Research_Paper.pdf",
                            mime="application/pdf",
                            type="primary",
                            use_container_width=True,
                        )
                else:
                    st.caption("Research Paper PDF not generated yet.")

        with col_doc2:
            with st.container(border=True):
                st.markdown("#### Comprehensive Project Report (PDF)")
                st.write("Complete institutional project report covering architectural design, multi-dataset pipeline, and hardware profiling.")
                if pdf_report_path.exists():
                    with open(pdf_report_path, "rb") as f_rep:
                        st.download_button(
                            "Download Project Report (PDF)",
                            data=f_rep.read(),
                            file_name="Quantum_Voice_Classification_Project_Report.pdf",
                            mime="application/pdf",
                            use_container_width=True,
                        )
                else:
                    st.caption("Project Report PDF not generated yet.")

        # Data Export Buttons
        with st.container(border=True):
            st.markdown("#### Empirical Benchmark Datasets & Artifacts")
            c_exp1, c_exp2, c_exp3 = st.columns(3)
            json_path = config.RESULTS_DIR / "evaluation_metrics.json"
            csv_path = config.RESULTS_DIR / "evaluation_metrics.csv"
            pred_path = config.RESULTS_DIR / "test_predictions.csv"

            with c_exp1:
                if json_path.exists():
                    with open(json_path, "r", encoding="utf-8") as f:
                        st.download_button(
                            "Download Metrics (JSON)",
                            data=f.read(),
                            file_name="quantum_benchmark_metrics.json",
                            mime="application/json",
                            use_container_width=True,
                        )
            with c_exp2:
                if csv_path.exists():
                    with open(csv_path, "r", encoding="utf-8") as f:
                        st.download_button(
                            "Download Metrics (CSV)",
                            data=f.read(),
                            file_name="quantum_benchmark_metrics.csv",
                            mime="text/csv",
                            use_container_width=True,
                        )
            with c_exp3:
                if pred_path.exists():
                    with open(pred_path, "r", encoding="utf-8") as f:
                        st.download_button(
                            "Download Test Predictions (CSV)",
                            data=f.read(),
                            file_name="test_predictions_multidataset.csv",
                            mime="text/csv",
                            use_container_width=True,
                        )

            # In-App Data Preview
            if json_path.exists():
                with st.expander("Preview Benchmark Metrics JSON Content", expanded=False):
                    with open(json_path, "r", encoding="utf-8") as f:
                        st.json(json.load(f))

        # Scientific Methodology Cards
        with st.container(border=True):
            st.markdown("#### Research Methodology & Scientific Invariants")
            st.markdown(r"""
            1. **Multi-Dataset Corpus Engineering:**
               - To evaluate domain shift, the corpus integrates **ASVspoof 2019 LA** (logical access attacks), **Kaggle Voices** (modern synthetic non-human generation), and **Mozilla Common Voice Indian Accent** (cross-demographic human speech).
               - All audio files are standardized to mono 16 kHz and fixed 3.0-second temporal windows with zero-padding or central cropping.

            2. **Acoustic Feature Extraction:**
               - 70 time-frequency descriptors extracted via Librosa: 40 MFCCs (mean/std), 2 Spectral Centroids, 2 Spectral Bandwidths, 2 Zero-Crossing Rates, and 24 Chroma STFT coefficients.
               - Fast SHA-256 disk caching prevents redundant signal processing during development and testing.

            3. **Zero Data Leakage Protocol:**
               - Preprocessing transformations (`StandardScaler` and `PCA`) are fitted exclusively on the training split $\mathcal{D}_{\text{train}}$ (80%).
               - Test data $\mathcal{D}_{\text{test}}$ (20%) is transformed using frozen scalers and projections.

            4. **Variational Quantum Classification (VQC):**
               - PCA projects 70 features into 4 orthogonal components mapped into single-qubit rotations $[0, \pi]$.
               - A $ZZ$ phase feature map introduces non-linear quantum entanglement.
               - Parameterized $R_y$ ansatz layers with CNOT entanglement ladders are optimized using COBYLA on Qiskit Aer statevector simulators.

            5. **Empirical Quantum Advantage Reality Assessment:**
               - Classical MLPs process single-sample inference in **~1-3 ms** on CPU/GPU.
               - Quantum statevector simulation on classical computers requires **~20-50 ms** due to exponential $2^n$ statevector calculation.
               - Quantum VQC operates in Hilbert space $\mathcal{H} = \mathbb{C}^{16}$ with only **12 variational angles**, demonstrating high parameter efficiency compared to deep neural networks.
            """)


if __name__ == "__main__":
    main()
