"""Streamlit Web Application: Quantum & Classical ML Voice Anti-Spoofing Classifier."""

import os
import json
import time
from pathlib import Path
from typing import Dict, Any, Optional

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
    load_dataset_features,
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
    page_title="Quantum Audio Forensics: AI vs Human Voice",
    page_icon="🎙️",
    layout="wide",
    initial_sidebar_state="expanded",
)


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


def main():
    st.title("🎙️ Quantum Audio Forensics: AI vs Human Voice")
    st.markdown(
        "**Comparative Analysis of Classical PyTorch Deep Learning (CPU vs NVIDIA CUDA) "
        "and Variational Quantum Classifiers (VQC on Qiskit Aer) using the ASVspoof 2019 LA Dataset.**"
    )

    models_bundle = load_system_models()
    devices = models_bundle["devices"]

    # Sidebar Hardware & Model Status
    st.sidebar.header("⚙️ System Status")
    st.sidebar.metric(
        label="NVIDIA CUDA GPU",
        value="Available" if devices["cuda"] else "Not Detected",
        delta=devices["gpu_name"] if devices["cuda"] else "CPU Only",
    )
    st.sidebar.metric(
        label="Quantum Simulator",
        value="Qiskit Aer",
        delta=f"{config.N_QUBITS} Qubits (PCA-4)",
    )

    cpu_status = "✅ Ready" if models_bundle["mlp_cpu"] else "⚠️ Needs Training"
    gpu_status = "✅ Ready" if models_bundle["mlp_gpu"] else ("⚠️ CPU Fallback" if not devices["cuda"] else "⚠️ Needs Training")
    q_status = "✅ Ready" if models_bundle["quantum_vqc"] else "⚠️ Needs Training"

    st.sidebar.markdown(f"""
    **Model Checkpoints:**
    - PyTorch CPU MLP: {cpu_status}
    - PyTorch GPU MLP: {gpu_status}
    - Qiskit VQC: {q_status}
    """)

    # Main Tabs
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "🔬 Audio Classification (Inference)",
        "⚛️ Quantum Circuit & Qubits",
        "📊 Evaluation & Benchmarks",
        "📁 Dataset & Acoustic Features",
        "💾 Results Export & Paper Methodology",
    ])

    # ----------------------------------------------------
    # TAB 1: AUDIO CLASSIFICATION & INFERENCE
    # ----------------------------------------------------
    with tab1:
        st.subheader("1. Audio Input & Signal Analysis")
        col_input1, col_input2 = st.columns([1, 1])

        audio_bytes = None
        source_label = "Custom Upload"

        with col_input1:
            uploaded_file = st.file_uploader(
                "Upload Voice Audio File (.flac, .wav, .mp3, .ogg)",
                type=["flac", "wav", "mp3", "ogg"],
            )

        with col_input2:
            st.markdown("**Or select pre-loaded ASVspoof 2019 samples:**")
            demo_files = list(config.DEMO_DIR.glob("*.flac"))
            demo_options = ["None (Use uploaded file)"] + [f.name for f in demo_files]
            selected_demo = st.selectbox("Choose benchmark demo sample:", demo_options)

            if selected_demo != "None (Use uploaded file)":
                chosen_path = config.DEMO_DIR / selected_demo
                with open(chosen_path, "rb") as f:
                    audio_bytes = f.read()
                source_label = f"Demo: {selected_demo}"
            elif uploaded_file is not None:
                audio_bytes = uploaded_file.read()
                source_label = f"Uploaded: {uploaded_file.name}"

        if audio_bytes is not None:
            st.markdown(f"**Loaded Audio:** `{source_label}`")
            st.audio(audio_bytes)

            # Signal waveform & spectrogram
            try:
                y, sr = load_audio(audio_bytes, sr=config.SAMPLE_RATE, duration=config.DURATION)
                with st.expander("📈 View Audio Waveform & Mel-Spectrogram", expanded=True):
                    fig_audio = plot_audio_analysis(y, sr)
                    st.pyplot(fig_audio)
                    plt.close(fig_audio)

                # Feature extraction
                feat_raw, meta = extract_features_with_cache(audio_bytes, use_cache=True)
                components = meta["components"]

                with st.expander("🎼 Acoustic Features Breakdown (70 Dimensions)", expanded=False):
                    c1, c2, c3 = st.columns(3)
                    with c1:
                        st.markdown("**MFCC (20 Coefficients):**")
                        st.dataframe(pd.DataFrame({
                            "Mean": components["mfcc_mean"].round(3),
                            "Std": components["mfcc_std"].round(3),
                        }), height=200)
                    with c2:
                        st.markdown("**Spectral Properties:**")
                        st.write(f"- Centroid Mean: `{components['centroid_mean'][0]:.2f} Hz`")
                        st.write(f"- Centroid Std: `{components['centroid_std'][0]:.2f} Hz`")
                        st.write(f"- Bandwidth Mean: `{components['bandwidth_mean'][0]:.2f} Hz`")
                        st.write(f"- Bandwidth Std: `{components['bandwidth_std'][0]:.2f} Hz`")
                        st.write(f"- Zero Crossing Rate: `{components['zcr_mean'][0]:.4f}`")
                    with c3:
                        st.markdown("**Chroma STFT (12 Pitch Classes):**")
                        st.dataframe(pd.DataFrame({
                            "Chroma Mean": components["chroma_mean"].round(3),
                            "Chroma Std": components["chroma_std"].round(3),
                        }), height=200)

                # Classification Section
                st.subheader("2. Triple Model Classification (CPU vs GPU vs Quantum)")

                if not PREPROCESSORS_PATH.exists():
                    st.error("Preprocessing pipeline not found. Please run the model training first.")
                else:
                    feat_scaled, feat_quantum = preprocess_single_sample(feat_raw)

                    col_m1, col_m2, col_m3 = st.columns(3)

                    # Model 1: Classical CPU MLP
                    with col_m1:
                        st.markdown("### 🖥️ Classical MLP (CPU)")
                        if models_bundle["mlp_cpu"] is not None:
                            pred_cpu = predict_single_sample(models_bundle["mlp_cpu"], feat_scaled, device_name="cpu")
                            is_spoof = pred_cpu["predicted_label"] == config.LABEL_SPOOF
                            st.metric(
                                label="Predicted Origin",
                                value="🤖 AI-Generated (Spoof)" if is_spoof else "👤 Human (Bonafide)",
                                delta=f"{pred_cpu['confidence']*100:.1f}% Confidence",
                            )
                            st.write(f"⏱️ **Inference Latency:** `{pred_cpu['latency_ms']:.2f} ms`")
                            st.progress(pred_cpu["confidence"])
                            st.caption(f"Probabilities: Human: {pred_cpu['probabilities'][config.LABEL_NAMES[0]]:.3f} | AI: {pred_cpu['probabilities'][config.LABEL_NAMES[1]]:.3f}")
                        else:
                            st.info("CPU MLP weights not found. Train models first.")

                    # Model 2: Classical GPU MLP
                    with col_m2:
                        st.markdown(f"### ⚡ Classical MLP (CUDA GPU)")
                        if models_bundle["mlp_gpu"] is not None:
                            pred_gpu = predict_single_sample(models_bundle["mlp_gpu"], feat_scaled, device_name="cuda")
                            is_spoof = pred_gpu["predicted_label"] == config.LABEL_SPOOF
                            st.metric(
                                label="Predicted Origin",
                                value="🤖 AI-Generated (Spoof)" if is_spoof else "👤 Human (Bonafide)",
                                delta=f"{pred_gpu['confidence']*100:.1f}% Confidence",
                            )
                            st.write(f"⏱️ **Inference Latency:** `{pred_gpu['latency_ms']:.2f} ms`")
                            st.progress(pred_gpu["confidence"])
                            st.caption(f"Probabilities: Human: {pred_gpu['probabilities'][config.LABEL_NAMES[0]]:.3f} | AI: {pred_gpu['probabilities'][config.LABEL_NAMES[1]]:.3f}")
                        elif models_bundle["mlp_cpu"] is not None and not devices["cuda"]:
                            st.warning("NVIDIA CUDA GPU not detected on this system. Showing CPU fallback.")
                        else:
                            st.info("GPU MLP weights not found. Train models first.")

                    # Model 3: Quantum VQC
                    with col_m3:
                        st.markdown("### ⚛️ Quantum VQC (Qiskit Aer)")
                        if models_bundle["quantum_vqc"] is not None:
                            pred_q = models_bundle["quantum_vqc"].predict_single(feat_quantum)
                            is_spoof = pred_q["predicted_label"] == config.LABEL_SPOOF
                            st.metric(
                                label="Predicted Origin",
                                value="🤖 AI-Generated (Spoof)" if is_spoof else "👤 Human (Bonafide)",
                                delta=f"{pred_q['confidence']*100:.1f}% Confidence",
                            )
                            st.write(f"⏱️ **Inference Latency:** `{pred_q['latency_ms']:.2f} ms`")
                            st.progress(pred_q["confidence"])
                            st.caption(f"Probabilities: Human: {pred_q['probabilities'][config.LABEL_NAMES[0]]:.3f} | AI: {pred_q['probabilities'][config.LABEL_NAMES[1]]:.3f}")
                        else:
                            st.info("Quantum VQC weights not found. Train models first.")

            except Exception as ex:
                st.error(f"Error processing audio sample: {ex}")
                import traceback
                st.code(traceback.format_exc())
        else:
            st.info("👆 Please upload an audio file or select a pre-loaded sample above to begin.")

    # ----------------------------------------------------
    # TAB 2: QUANTUM CIRCUIT & QUBITS
    # ----------------------------------------------------
    with tab2:
        st.subheader("⚛️ Variational Quantum Circuit (VQC) Architecture")
        st.markdown("""
        The quantum classifier operates on **4 qubits** initialized in state $|0000\\rangle$.
        1. **Dimensionality Reduction:** 70 acoustic features $\\to$ 4 principal components via PCA.
        2. **Quantum Feature Map:** Features are mapped into single-qubit rotation angles $\\theta_i \\in [0, \\pi]$ followed by entangling gates.
        3. **Parameterized Ansatz:** Trainable $R_y$ rotation gates followed by a CNOT entangling ladder.
        4. **Measurement:** Qubit expectation values / bitstring probabilities are sampled via **Qiskit Aer Simulator**.
        """)

        col_q1, col_q2 = st.columns([1, 1])
        with col_q1:
            fm_choice = st.selectbox(
                "Feature Map Type:",
                ["ZZFeatureMap", "CustomRotation"],
                index=0,
                help="ZZFeatureMap provides non-linear 2nd order phase entanglement. CustomRotation uses explicit Ry-CNOT-Rz rotations.",
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

        st.markdown(f"**Circuit Complexity:** Depth = `{full_qc.depth()}`, Total Gates = `{full_qc.size()}`, Trainable Parameters = `{ansatz_qc.num_parameters}`")

        c_draw1, c_draw2 = st.columns([2, 1])
        with c_draw1:
            st.markdown("**Quantum Circuit Diagram (Matplotlib Render):**")
            fig_qc = full_qc.draw(output="mpl")
            st.pyplot(fig_qc)
            plt.close(fig_qc)

        with c_draw2:
            st.markdown("**Quantum Circuit Diagram (Text Form):**")
            st.text(str(full_qc.draw(output="text")))

        st.markdown("### 🌐 Mathematical Formulation")
        st.latex(r"|\psi(x, \theta)\rangle = U_{\text{Ansatz}}(\theta) \cdot U_{\Phi}(x) |0\rangle^{\otimes 4}")
        st.latex(r"P(y=0|x, \theta) = \sum_{z \in \text{Even Parity}} |\langle z | \psi(x, \theta) \rangle|^2")

    # ----------------------------------------------------
    # TAB 3: EVALUATION & BENCHMARKS
    # ----------------------------------------------------
    with tab3:
        st.subheader("📊 Empirical Performance Comparison on Identical Held-Out Test Set")
        benchmark_results = load_benchmark_results()

        if benchmark_results:
            st.markdown("### 🏆 Metrics Summary Table")
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
                st.markdown("#### Performance Metrics")
                fig_m = plot_metrics_comparison(benchmark_results)
                st.pyplot(fig_m)
                plt.close(fig_m)

            with col_p2:
                st.markdown("#### Training & Inference Latency")
                fig_l = plot_latency_and_training_comparison(benchmark_results)
                st.pyplot(fig_l)
                plt.close(fig_l)

            st.markdown("### 🔲 Confusion Matrices")
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
        else:
            st.info("No benchmark results found. Run `python train_and_evaluate.py` to generate real empirical metrics.")

    # ----------------------------------------------------
    # TAB 4: DATASET & ACOUSTIC FEATURES
    # ----------------------------------------------------
    with tab4:
        st.subheader("📁 ASVspoof 2019 LA Dataset Partition & Features")
        st.info("🌐 **Hugging Face Benchmark Dataset URL:** [https://huggingface.co/datasets/SpeechAntiSpoofingBenchmarks/ASVspoof2019_LA](https://huggingface.co/datasets/SpeechAntiSpoofingBenchmarks/ASVspoof2019_LA) | **Official Upstream:** Edinburgh DataShare (doi:10.7488/ds/298)")
        if meta_path.exists():
            with open(meta_path, "r", encoding="utf-8") as f:
                records = json.load(f)
            df_meta = pd.DataFrame(records)

            d_col1, d_col2, d_col3 = st.columns(3)
            d_col1.metric("Total Audio Samples", len(df_meta))
            d_col2.metric("Bonafide (Human Speech)", len(df_meta[df_meta["label"] == config.LABEL_BONAFIDE]))
            d_col3.metric("Spoof (AI Synthesized)", len(df_meta[df_meta["label"] == config.LABEL_SPOOF]))

            st.markdown("#### Sample Metadata Browser")
            st.dataframe(df_meta[["utterance_id", "speaker_id", "label_name", "filename"]].head(20), use_container_width=True)

        if PREPROCESSORS_PATH.exists():
            import joblib
            pipe = joblib.load(PREPROCESSORS_PATH)
            st.markdown("#### PCA Explained Variance Ratio (4 Quantum Dimensions)")
            var_ratios = pipe.get("explained_variance_ratio", [])
            df_pca = pd.DataFrame({
                "Principal Component": [f"PC{i+1}" for i in range(len(var_ratios))],
                "Explained Variance Ratio": var_ratios,
                "Cumulative Variance": np.cumsum(var_ratios),
            })
            st.table(df_pca)
            st.caption(f"Total variance retained by 4 qubits: **{pipe.get('cumulative_variance', 0)*100:.2f}%**")

    # ----------------------------------------------------
    # TAB 5: RESULTS EXPORT & METHODOLOGY
    # ----------------------------------------------------
    with tab5:
        st.subheader("💾 Export Benchmark Results & Research Methodology")

        pdf_paper_path = config.BASE_DIR / "Quantum_Voice_Classification_Research_Paper.pdf"
        if pdf_paper_path.exists():
            with open(pdf_paper_path, "rb") as f_pdf:
                st.download_button(
                    "📄 Download Publication-Grade Research Paper (PDF)",
                    data=f_pdf.read(),
                    file_name="Quantum_Voice_Classification_Research_Paper.pdf",
                    mime="application/pdf",
                    type="primary",
                    use_container_width=True,
                )
            st.markdown("<br>", unsafe_allow_html=True)

        c_exp1, c_exp2, c_exp3 = st.columns(3)
        json_path = config.RESULTS_DIR / "evaluation_metrics.json"
        csv_path = config.RESULTS_DIR / "evaluation_metrics.csv"
        pred_path = config.RESULTS_DIR / "test_predictions.csv"

        with c_exp1:
            if json_path.exists():
                with open(json_path, "r", encoding="utf-8") as f:
                    st.download_button(
                        "📥 Download Metrics JSON",
                        data=f.read(),
                        file_name="asvspoof_quantum_benchmark.json",
                        mime="application/json",
                    )
        with c_exp2:
            if csv_path.exists():
                with open(csv_path, "r", encoding="utf-8") as f:
                    st.download_button(
                        "📥 Download Metrics CSV",
                        data=f.read(),
                        file_name="asvspoof_quantum_benchmark.csv",
                        mime="text/csv",
                    )
        with c_exp3:
            if pred_path.exists():
                with open(pred_path, "r", encoding="utf-8") as f:
                    st.download_button(
                        "📥 Download Predictions CSV",
                        data=f.read(),
                        file_name="test_set_predictions.csv",
                        mime="text/csv",
                    )

        st.markdown("---")
        st.markdown(r"""
        ### 📄 Research Methodology & Scientific Observations
        1. **Feature Engineering Invariant:**
           - Librosa extracts 70 time-frequency descriptors: 40 MFCCs (mean/std), 2 Spectral Centroids, 2 Spectral Bandwidths, 2 Zero-Crossing Rates, and 24 Chroma coefficients.
           - Audio is resampled to mono 16 kHz and zero-padded/cropped to 3.0 seconds.

        2. **Prevention of Data Leakage:**
           - Preprocessing transformations (`StandardScaler` and `PCA`) are fitted exclusively on the training split $\\mathcal{D}_{\\text{train}}$.
           - Test data $\\mathcal{D}_{\\text{test}}$ is transformed using the frozen preprocessors.

        3. **Variational Quantum Classification (VQC):**
           - Reduced to 4 dimensions via PCA and mapped onto 4 qubits using $ZZ$ phase entanglement.
           - Entanglement ladder of CNOT gates enables quantum correlations between acoustic components.
           - Evaluated using Qiskit Aer's statevector and shot-based simulation.

        4. **Quantum Advantage Reality Assessment:**
           - Simulation of quantum circuits on classical hardware (CPU/GPU) introduces exponential statevector overhead ($\mathcal{O}(2^n)$).
           - Classical MLPs remain orders of magnitude faster in latency on classical computers.
           - Potential quantum advantages on NISQ or fault-tolerant hardware lie in representational expressibility within high-dimensional Hilbert spaces ($\mathcal{H} = \mathbb{C}^{16}$ for 4 qubits) with few parameters.
        """)


if __name__ == "__main__":
    main()
