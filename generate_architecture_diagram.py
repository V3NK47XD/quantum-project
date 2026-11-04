"""Generate publication-quality system architecture diagram for Quantum Audio Forensics Hub."""

import matplotlib.pyplot as plt
import matplotlib.patches as patches
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
RESULTS_DIR = BASE_DIR / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_IMG = RESULTS_DIR / "architecture_diagram.png"


def create_architecture_diagram():
    fig, ax = plt.subplots(figsize=(14, 9), dpi=300)
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)
    ax.axis("off")

    # Font settings
    title_font = {"fontsize": 13, "fontweight": "bold", "color": "#0f172a"}
    section_font = {"fontsize": 10, "fontweight": "bold", "color": "#1e293b"}
    text_font = {"fontsize": 8.5, "color": "#334155"}
    code_font = {"fontsize": 8, "fontfamily": "monospace", "color": "#0f172a"}

    # Helper function to draw rounded boxes
    def draw_box(x, y, w, h, title, subtitle="", items=None, bg="#f8fafc", edge="#475569", lw=1.5):
        box = patches.FancyBboxPatch(
            (x, y), w, h,
            boxstyle="round,pad=0.8,rounding_size=2.0",
            facecolor=bg,
            edgecolor=edge,
            linewidth=lw,
            zorder=2,
        )
        ax.add_patch(box)
        
        # Header banner inside box
        header_y = y + h - 4.5
        ax.text(x + w / 2, header_y, title, fontdict=section_font, ha="center", va="center", zorder=3)
        if subtitle:
            ax.text(x + w / 2, header_y - 3.2, subtitle, fontdict={"fontsize": 7.5, "fontstyle": "italic", "color": "#64748b"}, ha="center", va="center", zorder=3)
        
        # Divider line
        div_y = header_y - (5.2 if subtitle else 3.2)
        ax.plot([x + 2, x + w - 2], [div_y, div_y], color="#cbd5e1", linewidth=0.8, zorder=3)
        
        # Content items
        if items:
            start_y = div_y - 3.2
            for i, itm in enumerate(items):
                ax.text(x + 3, start_y - (i * 3.4), itm, fontdict=text_font, ha="left", va="center", zorder=3)

    # 1. Tier 1: Multi-Dataset Sourcing & Audio Ingestion
    tier1_items = [
        "• ASVspoof 2019 LA Benchmark (100 Bonafide, 100 Spoof .flac)",
        "• Kaggle Human & Non-human Voices (100 Human, 200 Spoof .mp3)",
        "• CommonVoice Indian Accent (100 Human Speech .mp3)",
        "• Live Browser Microphone Audio Stream (Streamlit st.audio_input)",
        "• Total Corpus: 600 Audio Files (Strict 50.0% / 50.0% Class Balance)",
    ]
    draw_box(4, 76, 92, 21, "TIER 1: MULTI-DATASET SOURCING & AUDIO INGESTION", "Diverse Acoustic Repositories & Real-Time Browser Microphone Capture", tier1_items)

    # Down Arrow from Tier 1 to Tier 2
    ax.annotate(
        "", xy=(50, 71.5), xytext=(50, 76),
        arrowprops=dict(facecolor="#334155", edgecolor="#334155", width=2, headwidth=7, headlength=6),
        zorder=5
    )

    # 2. Tier 2: Acoustic Signal Processing & Feature Extraction
    tier2_items = [
        "• Preprocessing: Peak Normalization to [-1, 1], 16 kHz Resampling, 3.0s Fixed Window (N = 48,000)",
        "• 40 MFCC Statistics: Mean & Std Dev across 20 Mel-scale filterbank cepstral coefficients",
        "• Spectral Properties: Spectral Centroid (2D), Spectral Bandwidth (2D), Zero-Crossing Rate (2D)",
        "• Chroma Energy: Chroma STFT Mean & Std Dev across 12 pitch semitones (24D)  ==> Total: 70 Dimensions",
        "• Transparent Caching: SHA-256 Content-Hash Lookup (data/cache/{hash}.npz) for sub-ms repeat retrieval",
    ]
    draw_box(4, 50.5, 92, 21, "TIER 2: ACOUSTIC SIGNAL PROCESSING & FEATURE EXTRACTION (LIBROSA)", "70-Dimensional Classical Statistical Acoustic Feature Engineering", tier2_items)

    # Down Arrow from Tier 2 to Tier 3
    ax.annotate(
        "", xy=(50, 46), xytext=(50, 50.5),
        arrowprops=dict(facecolor="#334155", edgecolor="#334155", width=2, headwidth=7, headlength=6),
        zorder=5
    )

    # 3. Tier 3: Zero-Leakage Preprocessing & Latent Mapping
    tier3_items = [
        "• Stratified 80/20 Train-Test Split: 480 Training Samples, 120 Held-Out Testing Samples",
        "• StandardScaler: μ_train and σ_train computed strictly on D_train, frozen for D_test and live inference",
        "• Principal Component Analysis: PCA(n_components=4) projects 70D acoustic features into 4D latent space",
        "• MinMaxScaler: Scales 4D PCA components into quantum rotation angles θ_i ∈ [0, π] for gate rotations",
    ]
    draw_box(4, 28, 92, 18, "TIER 3: ZERO-LEAKAGE PREPROCESSING & DIMENSIONALITY REDUCTION", "Strict Data Isolation & Quantum Latent Angle Normalization (models/preprocessors.joblib)", tier3_items)

    # Bifurcation arrows from Tier 3 to Tier 4 (Dual Models)
    # Left arrow to Classical MLP
    ax.annotate(
        "", xy=(26, 23.5), xytext=(40, 28),
        arrowprops=dict(facecolor="#1e3a8a", edgecolor="#1e3a8a", width=1.8, headwidth=6, headlength=5),
        zorder=5
    )
    # Right arrow to Quantum VQC
    ax.annotate(
        "", xy=(74, 23.5), xytext=(60, 28),
        arrowprops=dict(facecolor="#4c1d95", edgecolor="#4c1d95", width=1.8, headwidth=6, headlength=5),
        zorder=5
    )

    # 4A. Classical Model Box (Left)
    c_items = [
        "• Input: 70-Dimensional Scaled Feature Vector",
        "• Layer 1: Linear(70 → 64) + BatchNorm + ReLU + Dropout(0.25)",
        "• Layer 2: Linear(64 → 32) + BatchNorm + ReLU + Dropout(0.25)",
        "• Output: Linear(32 → 2) + Softmax Class Probabilities",
        "• Optimization: CrossEntropyLoss + Adam (lr=0.001, Epochs=40)",
        "• Model Complexity: 6,818 trainable weights and biases",
        "• Test Accuracy: 81.67% | F1: 0.8103 | Latency: 0.30 ms",
    ]
    draw_box(4, 2.5, 44, 21, "4A: CLASSICAL PYTORCH MLP", "Deep Neural Network Baseline (CPU / CUDA GPU)", c_items, bg="#f8fafc", edge="#1e3a8a", lw=1.5)

    # 4B. Quantum Model Box (Right)
    q_items = [
        "• Quantum Register: 4 Qubits initialized in ground state |0000⟩",
        "• Feature Map: ZZFeatureMap (reps=1, 2-qubit phase entanglement)",
        "• Variational Ansatz: RealAmplitudes with CNOT entangling ladder",
        "• Model Complexity: 12 continuous trainable variational angles",
        "• Measurement: Qiskit Aer SamplerV2 (1024 shots) → Even/Odd Parity",
        "• Optimization: Gradient-free COBYLA (maxiter=40 iterations)",
        "• Test Accuracy: 55.83% | F1: 0.5138 | Latency: 17.03 ms",
    ]
    draw_box(52, 2.5, 44, 21, "4B: VARIATIONAL QUANTUM CLASSIFIER (VQC)", "Qiskit Aer Quantum Circuit Simulation (H = C^16)", q_items, bg="#f8fafc", edge="#4c1d95", lw=1.5)

    plt.tight_layout()
    fig.savefig(OUTPUT_IMG, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Architecture diagram generated successfully at: {OUTPUT_IMG}")


if __name__ == "__main__":
    create_architecture_diagram()
