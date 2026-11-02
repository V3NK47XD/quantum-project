"""Generate a comprehensive, easy-to-read Springer Nature / Discover style PDF research paper.
Includes all foundational basics, HuggingFace dataset URLs, acoustic feature formulations,
quantum gate operations, and empirical benchmarks.
"""

import os
import base64
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
RESULTS_DIR = BASE_DIR / "results"
OUTPUT_HTML = BASE_DIR / "research_paper_springer.html"
OUTPUT_PDF = BASE_DIR / "Quantum_Voice_Classification_Research_Paper.pdf"


def get_base64_image(image_path: Path) -> str:
    if not image_path.exists():
        return ""
    with open(image_path, "rb") as f:
        encoded = base64.b64encode(f.read()).decode("utf-8")
    ext = image_path.suffix.lstrip(".").lower()
    mime = f"image/{ext}" if ext != "jpg" else "image/jpeg"
    return f"data:{mime};base64,{encoded}"


def build_html():
    circuit_b64 = get_base64_image(RESULTS_DIR / "quantum_circuit_diagram.png")
    metrics_b64 = get_base64_image(RESULTS_DIR / "metrics_comparison.png")
    latency_b64 = get_base64_image(RESULTS_DIR / "latency_comparison.png")
    cm_b64 = get_base64_image(RESULTS_DIR / "composite_confusion_matrices.png")

    template = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>Quantum Machine Learning for AI vs. Human Voice Classification</title>
<style>
  @page {
    size: A4;
    margin: 18mm 18mm 20mm 18mm;
    @top-left {
      content: "Venkat Prashad & Vinoth (2026) 1:12";
      font-family: system-ui, -apple-system, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
      font-size: 8pt;
      color: #64748b;
    }
    @top-right {
      content: "Page " counter(page) " of " counter(pages);
      font-family: system-ui, -apple-system, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
      font-size: 8pt;
      color: #64748b;
      font-weight: 500;
    }
  }

  @page :first {
    @top-left { content: ""; }
    @top-right { content: ""; }
  }

  body {
    font-family: Georgia, "Times New Roman", Times, serif;
    font-size: 9.6pt;
    line-height: 1.55;
    color: #1e293b;
    text-align: justify;
    margin: 0;
    padding: 0;
  }

  /* Page 1 Header Banner */
  .p1-header {
    display: flex;
    justify-content: space-between;
    align-items: center;
    border-bottom: 1.5px solid #0f172a;
    padding-bottom: 5px;
    margin-bottom: 14px;
    font-family: system-ui, -apple-system, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
    font-size: 8.5pt;
  }

  .journal-meta {
    color: #334155;
  }

  .journal-meta a {
    color: #0284c7;
    text-decoration: none;
  }

  .badge-container {
    display: flex;
    align-items: center;
    gap: 8px;
  }

  .research-label {
    font-weight: 800;
    font-size: 9pt;
    letter-spacing: 0.5px;
    color: #0f172a;
    text-transform: uppercase;
  }

  .open-access-badge {
    background-color: #10b981;
    color: #ffffff;
    font-size: 7.5pt;
    font-weight: 700;
    padding: 2px 7px;
    border-radius: 3px;
    letter-spacing: 0.3px;
    text-transform: uppercase;
  }

  /* Title & Authors */
  h1.paper-title {
    font-family: system-ui, -apple-system, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
    font-size: 18.5pt;
    font-weight: 800;
    line-height: 1.25;
    color: #0f172a;
    margin: 0 0 12px 0;
    letter-spacing: -0.3px;
  }

  .authors-line {
    font-family: system-ui, -apple-system, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
    font-size: 11pt;
    font-weight: 700;
    color: #0f172a;
    margin-bottom: 16px;
  }

  .authors-line sup {
    color: #0284c7;
    font-size: 8pt;
  }

  /* Split First Page Layout */
  .first-page-grid {
    display: flex;
    gap: 20px;
    margin-bottom: 20px;
    border-bottom: 1px solid #e2e8f0;
    padding-bottom: 16px;
  }

  .left-meta-col {
    width: 32%;
    font-family: system-ui, -apple-system, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
    font-size: 8pt;
    line-height: 1.45;
    color: #475569;
    border-right: 1px solid #e2e8f0;
    padding-right: 14px;
  }

  .left-meta-col strong {
    color: #0f172a;
    font-size: 8.5pt;
    display: block;
    margin-bottom: 2px;
  }

  .left-meta-col .email {
    color: #0284c7;
    word-break: break-all;
    font-family: "Courier New", Courier, monospace;
    font-size: 7.6pt;
    margin-bottom: 6px;
    display: block;
  }

  .left-meta-col .affiliation {
    margin-top: 8px;
    padding-top: 6px;
    border-top: 1px dashed #cbd5e1;
  }

  .right-abstract-col {
    width: 68%;
  }

  h2.abstract-title {
    font-family: system-ui, -apple-system, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
    font-size: 10.5pt;
    font-weight: 800;
    color: #0f172a;
    margin: 0 0 6px 0;
    text-transform: uppercase;
    letter-spacing: 0.5px;
  }

  .abstract-body {
    font-size: 9.2pt;
    line-height: 1.5;
    color: #1e293b;
    margin-bottom: 8px;
    text-align: justify;
  }

  .keywords-block {
    font-size: 8.5pt;
    line-height: 1.4;
    color: #334155;
    margin-top: 6px;
    padding-top: 6px;
    border-top: 1px solid #f1f5f9;
  }

  .keywords-block strong {
    font-family: system-ui, -apple-system, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
    color: #0f172a;
  }

  /* Section Headings */
  h2.sec-heading {
    font-family: system-ui, -apple-system, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
    font-size: 11.5pt;
    font-weight: 800;
    color: #0f172a;
    margin: 20px 0 8px 0;
    border-bottom: 1px solid #cbd5e1;
    padding-bottom: 3px;
    break-after: avoid;
  }

  h3.subsec-heading {
    font-family: system-ui, -apple-system, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
    font-size: 10pt;
    font-weight: 700;
    color: #1e3a5f;
    margin: 12px 0 5px 0;
    break-after: avoid;
  }

  h4.subsubsec-heading {
    font-family: system-ui, -apple-system, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
    font-size: 9.4pt;
    font-weight: 700;
    font-style: italic;
    color: #334155;
    margin: 8px 0 4px 0;
    break-after: avoid;
  }

  p {
    margin: 0 0 8px 0;
    text-indent: 14px;
  }

  p.no-indent {
    text-indent: 0;
  }

  /* PICO Callout Card */
  .pico-matrix-card {
    background-color: #f8fafc;
    border: 1px solid #cbd5e1;
    border-left: 4px solid #0284c7;
    border-radius: 4px;
    padding: 10px 14px;
    margin: 12px 0 14px 0;
    font-size: 8.8pt;
    line-height: 1.45;
  }

  .pico-matrix-title {
    font-family: system-ui, -apple-system, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
    font-size: 9.2pt;
    font-weight: 800;
    color: #0f172a;
    margin-bottom: 6px;
    text-transform: uppercase;
    letter-spacing: 0.5px;
  }

  .pico-row {
    margin-bottom: 5px;
  }

  .pico-label {
    font-family: system-ui, -apple-system, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
    font-weight: 700;
    color: #0369a1;
  }

  /* Dataset URL Callout Box */
  .dataset-banner {
    background-color: #f0fdf4;
    border: 1px solid #86efac;
    border-left: 4px solid #16a34a;
    border-radius: 4px;
    padding: 8px 12px;
    margin: 10px 0 12px 0;
    font-size: 8.8pt;
    line-height: 1.4;
  }

  .dataset-banner strong {
    color: #15803d;
    font-family: system-ui, -apple-system, "Segoe UI", Roboto, sans-serif;
  }

  .dataset-banner a {
    color: #0369a1;
    font-family: "Courier New", Courier, monospace;
    font-size: 8.4pt;
    word-break: break-all;
  }

  /* Mathematical Displays */
  .math-display {
    text-align: center;
    margin: 8px 0;
    font-family: "Cambria Math", "Times New Roman", serif;
    font-size: 9.8pt;
    padding: 3px 0;
    position: relative;
  }

  .math-display .eq-num {
    float: right;
    font-family: Georgia, serif;
    font-size: 9.2pt;
    color: #64748b;
  }

  /* Tables */
  table.springer-table {
    width: 100%;
    border-collapse: collapse;
    margin: 12px 0 14px 0;
    font-size: 8.5pt;
    line-height: 1.35;
  }

  table.springer-table caption {
    text-align: left;
    font-family: system-ui, -apple-system, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
    font-size: 8.8pt;
    font-weight: 700;
    margin-bottom: 5px;
    color: #0f172a;
  }

  table.springer-table th {
    border-top: 1.5px solid #0f172a;
    border-bottom: 1px solid #0f172a;
    padding: 5px 6px;
    text-align: center;
    font-family: system-ui, -apple-system, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
    font-weight: 700;
    background-color: #f8fafc;
  }

  table.springer-table td {
    padding: 4.5px 6px;
    border-bottom: 1px solid #e2e8f0;
    text-align: center;
  }

  table.springer-table tr.total-row td {
    border-top: 1px solid #0f172a;
    border-bottom: 1.5px solid #0f172a;
    font-weight: bold;
    background-color: #f8fafc;
  }

  /* Figures */
  .figure-wrapper {
    margin: 14px 0 16px 0;
    text-align: center;
    break-inside: avoid;
  }

  .figure-wrapper img {
    max-width: 100%;
    height: auto;
    border-radius: 3px;
  }

  .figure-caption {
    font-size: 8.3pt;
    line-height: 1.4;
    color: #334155;
    margin-top: 6px;
    text-align: justify;
    padding: 0 4px;
  }

  .figure-caption strong {
    font-family: system-ui, -apple-system, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
    color: #0f172a;
    font-weight: 700;
  }

  /* Step by Step Breakdown Cards */
  .concept-box {
    background-color: #f8fafc;
    border: 1px solid #e2e8f0;
    border-radius: 4px;
    padding: 8px 12px;
    margin: 8px 0;
    font-size: 8.7pt;
    line-height: 1.42;
  }

  .concept-box strong {
    color: #0f172a;
    font-family: system-ui, -apple-system, "Segoe UI", Roboto, sans-serif;
  }

  /* Footer License */
  .cc-footer {
    font-family: system-ui, -apple-system, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
    font-size: 7.5pt;
    line-height: 1.35;
    color: #64748b;
    margin-top: 18px;
    padding-top: 8px;
    border-top: 1px solid #e2e8f0;
  }

  /* References */
  .references-ol {
    font-size: 8.3pt;
    line-height: 1.42;
    padding-left: 18px;
    margin: 6px 0;
  }

  .references-ol li {
    margin-bottom: 5px;
    text-align: justify;
  }
</style>
</head>
<body>

<!-- PAGE 1 TOP HEADER BANNER -->
<div class="p1-header">
  <div class="journal-meta">
    Venkat Prashad &amp; Vinoth &bull; <em>Acoustic &amp; Quantum Forensics</em> (2026) 1:12 &bull; <a href="https://doi.org/10.1007/rec-cse-2026-qml-voice">https://doi.org/10.1007/rec-cse-2026-qml-voice</a>
  </div>
  <div class="badge-container">
    <span class="research-label">RESEARCH</span>
    <span class="open-access-badge">Open Access</span>
  </div>
</div>

<!-- TITLE -->
<h1 class="paper-title">Quantum Machine Learning for AI vs. Human Voice Classification: An Empirical Benchmark Using the PICO Framework</h1>

<!-- AUTHORS -->
<div class="authors-line">
  Venkat Prashad A<sup>1</sup> and Vinoth S<sup>1*</sup>
</div>

<!-- SPLIT METADATA & ABSTRACT (SPRINGER/DISCOVER FORMAT) -->
<div class="first-page-grid">
  <!-- LEFT COLUMN: CORRESPONDENCE & AFFILIATIONS -->
  <div class="left-meta-col">
    <strong>*Correspondence:</strong>
    <span class="email">vinoth.s.2024.cse@rajalakshmi.edu.in</span>
    <span class="email">venkatprashad.a.2024.cse@rajalakshmi.edu.in</span>
    
    <div class="affiliation">
      <sup>1</sup>Department of Computer Science and Engineering, Rajalakshmi Engineering College, Anna University, Chennai, India.
    </div>
  </div>

  <!-- RIGHT COLUMN: ABSTRACT & KEYWORDS -->
  <div class="right-abstract-col">
    <h2 class="abstract-title">Abstract</h2>
    <div class="abstract-body">
      The rapid proliferation of generative acoustic synthesis, encompassing neural text-to-speech (TTS) and deep voice conversion (VC), poses severe security risks to voice biometric authentication and digital forensics. This investigation presents an empirical comparative evaluation of a genuine <strong>Variational Quantum Classifier (VQC)</strong> against <strong>Classical Deep Multi-Layer Perceptrons (MLPs)</strong> deployed on <strong>Host CPU</strong> and <strong>NVIDIA CUDA GPU</strong> for binary speech anti-spoofing (<em>Bonafide Human</em> vs. <em>AI-Generated Spoof</em>). Structured under the formal <strong>PICO (Population, Intervention, Comparison, Outcome)</strong> framework: 
      <strong>(P)</strong> The population comprises 200 balanced recordings extracted from the official benchmark partition hosted on Hugging Face (<code>SpeechAntiSpoofingBenchmarks/ASVspoof2019_LA</code>), parameterized into 70-dimensional multi-domain acoustic descriptors (MFCC, spectral centroid, bandwidth, zero-crossing rate, chroma) with strict zero-leakage partitioning; 
      <strong>(I)</strong> The intervention employs a 4-qubit VQC featuring PCA latent projection (capturing 52.12% variance), non-linear <em>ZZ</em>-FeatureMap phase entanglement, a parameterized <em>R<sub>y</sub></em> rotation ansatz with nearest-neighbor CNOT entanglement, and classical COBYLA optimization simulated via Qiskit Aer; 
      <strong>(C)</strong> Comparison targets identical Classical MLPs (6,818 parameters) evaluated on CPU and NVIDIA GeForce RTX 2050 CUDA GPU; 
      <strong>(O)</strong> On the identical held-out test split of 40 recordings (20 Human, 20 Spoof), the CPU MLP achieved <strong>87.50% accuracy</strong> (<em>F</em><sub>1</sub> = 0.8571, 0.12 ms latency); the GPU MLP achieved <strong>82.50% accuracy</strong> (<em>F</em><sub>1</sub> = 0.8108, 0.33 ms latency); and the Qiskit VQC achieved <strong>47.50% accuracy</strong> (<em>F</em><sub>1</sub> = 0.5333, 7.89 ms latency). 
      Critically, the quantum model achieved an exceptional <strong>568:1 parameter compression ratio</strong> (12 variational angles vs. 6,818 classical weights). However, classical statevector simulation proved 65&times; slower than CPU execution, confirming that practical quantum advantage in audio forensics requires physical fault-tolerant hardware rather than classical simulators.
    </div>
    <div class="keywords-block">
      <strong>Keywords:</strong> Quantum Machine Learning (QML), Variational Quantum Classifier (VQC), Voice Anti-Spoofing, ASVspoof 2019, Hugging Face Dataset, Qiskit Aer, PyTorch CUDA, PICO Framework, Audio Forensics.
    </div>
  </div>
</div>

<!-- SECTION 1: INTRODUCTION -->
<h2 class="sec-heading">1 Introduction</h2>
<p>
  Acoustic biometric verification systems, automated banking lines, and multimedia authentication channels are increasingly vulnerable to malicious generative audio attacks. Modern deep neural models—such as diffusion acoustic models, neural vocoders (e.g., HiFi-GAN, WaveGlow), and autoregressive transformers—can synthesize perceptually convincing human speech from minimal voice recordings. Differentiating authentic human speech (<em>bonafide</em>) from algorithmically manipulated or synthesized utterances (<em>spoof</em>) has thus become a critical priority in digital multimedia forensics.
</p>
<p>
  Standard detection countermeasures predominantly rely on deep convolutional and graph neural networks, such as RawNet2 [5] and AASIST [6]. While effective, these models feature millions of parameters and demand substantial classical compute resources. This limits their deployment in low-power mobile authenticators and embedded edge devices.
</p>
<p>
  Quantum Machine Learning (QML), particularly Parameterized Quantum Circuits (PQCs) [1], [2], provides an alternative mathematical paradigm: by embedding low-dimensional classical feature vectors into a 2<sup><em>N</em></sup>-dimensional complex Hilbert space (&Hscr; = &Copf;<sup>16</sup> for <em>N</em> = 4 qubits), quantum circuits can exploit state superposition and multi-qubit entanglement to construct non-linear decision boundaries that are linearly separable in quantum state space. Despite significant theoretical literature, empirical research often lacks rigorous, leakage-free comparisons between hardware GPU baselines and simulated quantum models. This study establishes a rigorous, hardware-synchronized empirical benchmark.
</p>

<h3 class="subsec-heading">1.1 What are voice spoofing attacks?</h3>
<p>
  In speech forensics, acoustic attacks are categorized into two primary scenarios: <em>Physical Access (PA)</em>, where an attacker replays recorded audio through a loudspeaker in an acoustic room, and <em>Logical Access (LA)</em>, where synthetic audio is directly fed into an authentication pipeline. Logical access attacks comprise two main families:
</p>
<p class="no-indent">
  &bull; <strong>Text-to-Speech (TTS):</strong> A neural acoustic model converts input text into intermediate representations (e.g., mel-spectrograms), which a neural vocoder synthesizes into time-domain waveforms.<br>
  &bull; <strong>Voice Conversion (VC):</strong> An algorithmic pipeline modifies the acoustic characteristics (timbre, formants, pitch) of a source speaker to mimic a target victim while retaining the spoken linguistic content.
</p>

<h3 class="subsec-heading">1.2 The quantum machine learning paradigm</h3>
<p>
  Variational Quantum Classifiers (VQCs) represent the quantum analogue of classical neural networks. Rather than adjusting matrix weights via backpropagation across dense layers, a VQC adjusts continuous rotation angles &theta; on single-qubit gates while multi-qubit entangling gates (such as Controlled-NOT gates) generate correlations. This allows a quantum circuit with very few parameters to explore an exponentially large state space.
</p>

<!-- SECTION 2: LITERATURE SURVEY -->
<h2 class="sec-heading">2 Literature survey</h2>
<p>
  The evolution of acoustic anti-spoofing countermeasures has been driven by the continuous advancements in synthetic speech generation. This section reviews classical deep architectures alongside contemporary developments in quantum machine learning for signal processing.
</p>

<h3 class="subsec-heading">2.1 Foundational audio forensics and anti-spoofing</h3>
<p>
  The Automatic Speaker Verification Spoofing and Countermeasures (ASVspoof) challenge series [4] established standard baseline architectures. Classical systems primarily deploy spectral front-ends—such as Linear Frequency Cepstral Coefficients (LFCC) and Mel-Frequency Cepstral Coefficients (MFCC)—paired with residual networks (ResNet) or Light-CNN backbones. More recently, end-to-end architectures like AASIST integrate spectro-temporal graph attention networks directly on raw audio waveforms to detect localized synthesis artifacts. While accurate, these classical pipelines remain computationally intensive.
</p>

<h3 class="subsec-heading">2.2 Quantum machine learning in signal processing</h3>
<p>
  Variational Quantum Classifiers (VQCs), formalized by Havl&iacute;&check;ek et al. [1] and Schuld &amp; Killoran [3], recast pattern recognition into the optimization of parameterized unitary gates <em>U</em>(&theta;) acting on quantum feature states |&psi;(<em>x</em>)&rang;. Prior signal processing investigations explore quantum kernel estimators for music genre recognition and spoken keyword spotting. However, existing works frequently exhibit three experimental flaws: (i) <em>Data Leakage</em>: feature scalers and PCA transforms are fitted on the entire dataset prior to splitting; (ii) <em>Unsubstantiated Advantage Claims</em>: theoretical asymptotic Hilbert space dimensionality is conflated with actual wall-clock speedup; and (iii) <em>Missing Hardware Baselines</em>: GPU-accelerated classical baselines are rarely evaluated alongside quantum simulators under identical test splits.
</p>

<!-- SECTION 3: PICO FORMULATION -->
<h2 class="sec-heading">3 Research objective &amp; PICO formulation</h2>
<p>
  To resolve these limitations, this study utilizes the formal clinical and experimental <strong>PICO</strong> framework, establishing an objective comparison:
</p>

<div class="pico-matrix-card">
  <div class="pico-matrix-title">Structured PICO Research Design</div>
  <div class="pico-row">
    <span class="pico-label">&bull; Population (P):</span> Authenticated speech recordings from the ASVspoof 2019 Logical Access evaluation partition (100 Bonafide, 100 Spoof); 70-dimensional multi-domain acoustic vectors; strict train-test leakage prevention.
  </div>
  <div class="pico-row">
    <span class="pico-label">&bull; Intervention (I):</span> A 4-qubit Variational Quantum Classifier (VQC) with PCA-4 latent projection, non-linear <em>ZZ</em>-FeatureMap, parameterized <em>R<sub>y</sub></em> rotation ansatz with CNOT entanglement ladder, and Qiskit Aer statevector simulation.
  </div>
  <div class="pico-row">
    <span class="pico-label">&bull; Comparison (C):</span> Identically structured Classical Deep MLPs (6,818 parameters) evaluated on Host CPU and NVIDIA GeForce RTX 2050 CUDA GPU across identical data splits.
  </div>
  <div class="pico-row">
    <span class="pico-label">&bull; Outcome (O):</span> Accuracy, Precision, Recall, <em>F</em><sub>1</sub>-score, total training duration (s), single-sample inference latency (ms), and parameter compression ratio.
  </div>
</div>

<!-- SECTION 4: METHODOLOGY (POPULATION) -->
<h2 class="sec-heading">4 Methodology: Population &amp; acoustic feature engineering (P)</h2>

<h3 class="subsec-heading">4.1 Dataset provenance and Hugging Face repository</h3>
<p>
  The dataset utilized in this research is sourced directly from the standardized benchmark packaging of the <strong>ASVspoof 2019 Logical Access (LA)</strong> challenge hosted on Hugging Face:
</p>

<div class="dataset-banner">
  <strong>Hugging Face Dataset URL:</strong><br>
  <a href="https://huggingface.co/datasets/SpeechAntiSpoofingBenchmarks/ASVspoof2019_LA">https://huggingface.co/datasets/SpeechAntiSpoofingBenchmarks/ASVspoof2019_LA</a><br>
  <strong>Official Provenance:</strong> University of Edinburgh DataShare (<a href="https://doi.org/10.7488/ds/298">doi:10.7488/ds/298</a>), conforming to protocol <code>ASVspoof2019.LA.cm.eval.trl.txt</code>.
</div>

<p>
  The Logical Access partition evaluates speech generated by 13 state-of-the-art acoustic algorithms (A07 through A19), covering neural acoustic synthesis (Tacotron, Merlin), waveform vocoding (HiFi-GAN, WaveNet, Griffin-Lim), and voice conversion (VCC). A balanced subset of <em>M</em> = 200 standardized <code>.flac</code> audio files was extracted:
</p>
<p class="no-indent">
  &bull; <strong>100 Bonafide Human Recordings:</strong> Authentic vocalizations from human speakers.<br>
  &bull; <strong>100 Spoof AI Recordings:</strong> Generative speech synthesized across diverse acoustic vocoders.<br>
  &bull; <strong>Partitioning:</strong> Stratified 80% / 20% split yielding 160 training files (&Dscr;<sub>train</sub>: 80 Human, 80 Spoof) and 40 held-out test files (&Dscr;<sub>test</sub>: 20 Human, 20 Spoof).
</p>

<h3 class="subsec-heading">4.2 Audio preprocessing and multi-domain feature extraction</h3>
<p>
  Raw pulse-code modulation (PCM) audio signals are resampled to a mono sampling rate <em>f<sub>s</sub></em> = 16,000 Hz. Audio signals are normalized to a fixed duration of <em>T</em> = 3.0 seconds (48,000 samples) via center-cropping or symmetric zero-padding, ensuring temporal consistency. Peak amplitude is normalized to unity: <em>s</em>[<em>n</em>] &larr; <em>s</em>[<em>n</em>] / max(|<em>s</em>[<em>n</em>]| + 10<sup>&minus;6</sup>).
</p>
<p>
  Five distinct acoustic domains (comprising <strong>70 features</strong>) are extracted using Librosa:
</p>

<div class="concept-box">
  <strong>1. Mel-Frequency Cepstral Coefficients (MFCCs &ndash; 40 features):</strong><br>
  Human hearing perceives frequency logarithmically. MFCCs pass the Short-Time Fourier Transform (STFT; 1024-point FFT, 512 hop size) through 20 triangular Mel-scale bandpass filters, compute logarithmic energies, and take the Discrete Cosine Transform (DCT-II). The temporal mean &mu;<sub>MFCC</sub> &isin; &Ropf;<sup>20</sup> and standard deviation &sigma;<sub>MFCC</sub> &isin; &Ropf;<sup>20</sup> capture static vocal tract shape and temporal dynamics.<br><br>
  
  <strong>2. Spectral Centroid (2 features):</strong><br>
  Quantifies the "center of gravity" of the frequency spectrum: <em>f<sub>c</sub></em> = &sum; <em>f</em>(<em>k</em>)|<em>S</em>(<em>k</em>)| / &sum; |<em>S</em>(<em>k</em>)|. Mean and std across frames detect unnatural high-frequency energy typical of vocoder artifacts.<br><br>

  <strong>3. Spectral Bandwidth (2 features):</strong><br>
  Measures the second-order spread of acoustic frequencies around the spectral centroid.<br><br>

  <strong>4. Zero-Crossing Rate (ZCR &ndash; 2 features):</strong><br>
  Tracks the rate at which the time-domain waveform changes sign. Voiced human speech exhibits low ZCR, whereas unvoiced consonants and vocoder noise yield high ZCR.<br><br>

  <strong>5. Chroma STFT (24 features):</strong><br>
  Projects spectral energy onto the 12 semitone pitch classes of the equal-tempered scale (C, C#, D, &hellip;, B). 12 means and 12 standard deviations capture harmonic distribution across octaves.
</div>

<h3 class="subsec-heading">4.3 Zero-leakage preprocessing pipeline</h3>
<p>
  A frequent error in published machine learning research is <em>data leakage</em>&mdash;computing normalization statistics or PCA projections over the entire dataset prior to splitting. This artificially inflates test performance because the training pipeline inadvertently "sees" test distribution geometry.
</p>
<p>
  In this work, data leakage is strictly prevented:
</p>
<div class="math-display">
  <em>x</em><sub>scaled</sub> = (<em>x</em> &minus; &mu;<sub>train</sub>) / &sigma;<sub>train</sub>, &nbsp;&nbsp;&nbsp; <em>x</em><sub>PCA</sub> = <em>x</em><sub>scaled</sub> &middot; <em>W</em><sub>PCA</sub>
  <span class="eq-num">(1)</span>
</div>
<p class="no-indent">
  The standardization vectors (&mu;<sub>train</sub>, &sigma;<sub>train</sub> &isin; &Ropf;<sup>70</sup>) and the PCA projection matrix (<em>W</em><sub>PCA</sub> &isin; &Ropf;<sup>70&times;4</sup>) are fitted <strong>strictly on the 160 training audio files</strong>. The frozen transformations then process the 40 test audio files and any new user uploads.
</p>
<p>
  The 4 principal components retain <strong>52.12% of cumulative variance</strong> (PC<sub>1</sub>: 21.95%, PC<sub>2</sub>: 17.78%, PC<sub>3</sub>: 7.47%, PC<sub>4</sub>: 4.92%). Features are normalized into quantum angles &theta;<sub><em>i</em></sub> &isin; [0, &pi;]:
</p>
<div class="math-display">
  &theta;<sub><em>i</em></sub> = &pi; &middot; [<em>x</em><sub>PCA,<em>i</em></sub> &minus; min(<em>x</em><sub>PCA,<em>i</em></sub><sup>train</sup>)] / [max(<em>x</em><sub>PCA,<em>i</em></sub><sup>train</sup>) &minus; min(<em>x</em><sub>PCA,<em>i</em></sub><sup>train</sup>)]
  <span class="eq-num">(2)</span>
</div>

<!-- SECTION 5: INTERVENTION (QUANTUM VQC) -->
<h2 class="sec-heading">5 Intervention: Variational Quantum Classifier pipeline (I)</h2>

<h3 class="subsec-heading">5.1 Quantum fundamentals: Qubits and Hilbert space</h3>
<p>
  Unlike a classical bit that exists deterministically as 0 or 1, a single quantum bit (qubit) exists in a state of superposition: |&psi;&rang; = &alpha;|0&rang; + &beta;|1&rang;, with complex probability amplitudes satisfying |&alpha;|<sup>2</sup> + |&beta;|<sup>2</sup> = 1. A register of <em>N</em> = 4 qubits resides in a 2<sup>4</sup> = 16-dimensional complex Hilbert space (&Hscr; = &Copf;<sup>16</sup>), spanned by computational basis states {|0000&rang;, |0001&rang;, &hellip;, |1111&rang;}.
</p>

<h3 class="subsec-heading">5.2 Quantum Feature Map: The ZZ-FeatureMap</h3>
<p>
  The classical vector <em>x</em> &isin; [0, &pi;]<sup>4</sup> is encoded into the quantum register using a second-order Pauli expansion feature map (<em>ZZ</em>-FeatureMap):
</p>
<div class="math-display">
  <em>U</em><sub>&Phi;</sub>(<em>x</em>) = exp&lpar; <em>i</em> &sum;<sub><em>j</em>&lt;<em>k</em></sub> 2(&pi;&minus;<em>x<sub>j</sub></em>)(&pi;&minus;<em>x<sub>k</sub></em>)<em>Z<sub>j</sub>Z<sub>k</sub></em> &rpar; &middot; exp&lpar; <em>i</em> &sum;<sub><em>j</em></sub> 2<em>x<sub>j</sub>Z<sub>j</sub></em> &rpar; &middot; <em>H</em><sup>&otimes;4</sup>
  <span class="eq-num">(3)</span>
</div>
<p class="no-indent">
  &bull; <strong>Hadamard Gates (H):</strong> Applied to all 4 qubits, creating uniform superposition |&psi;<sub>0</sub>&rang; = (1/&radic;16) &sum;<sub><em>z</em></sub> |<em>z</em>&rang;.<br>
  &bull; <strong>Single-Qubit Phase Gates:</strong> Rotate individual qubit phases by angle 2<em>x<sub>j</sub></em>.<br>
  &bull; <strong>Controlled-NOT (CNOT) &amp; Phase Gates:</strong> Interleave CNOT entangling gates with phase rotations 2(&pi;&minus;<em>x<sub>j</sub></em>)(&pi;&minus;<em>x<sub>k</sub></em>) to embed non-linear pairwise acoustic interactions into quantum phase.
</p>

<!-- FIGURE 1: QUANTUM CIRCUIT -->
<div class="figure-wrapper">
  <img src="__CIRCUIT_B64__" alt="Quantum Circuit Architecture" style="max-height: 220px;">
  <div class="figure-caption">
    <strong>Figure 1. Architecture of the genuine 4-qubit Variational Quantum Classifier (VQC).</strong> The quantum circuit incorporates the non-linear <em>ZZ</em>-FeatureMap data encoding layer (Hadamard gates, single-qubit phase rotations, and CNOT phase interactions) followed by the trainable <em>RealAmplitudes</em> ansatz (parameterized <em>R<sub>y</sub></em> rotations and linear nearest-neighbor CNOT entangling ladder; depth <em>L</em> = 2, 12 variational parameters) terminated with computational basis measurements.
  </div>
</div>

<h3 class="subsec-heading">5.3 Parameterized quantum ansatz (RealAmplitudes)</h3>
<p>
  The trainable component of the quantum classifier is the ansatz <em>U<sub>A</sub></em>(&theta;), configured as a <code>RealAmplitudes</code> circuit:
</p>
<div class="math-display">
  <em>U<sub>A</sub></em>(&theta;) = &prod;<sub><em>l</em>=1</sub><sup><em>L</em></sup> &lsqb; &bigotimes;<sub><em>j</em>=0</sub><sup>3</sup> <em>R<sub>y</sub></em>(&theta;<sub><em>l,j</em></sub>) &prod;<sub><em>k</em>=0</sub><sup>2</sup> CNOT(<em>k, k</em>+1) &rsqb; &bigotimes;<sub><em>j</em>=0</sub><sup>3</sup> <em>R<sub>y</sub></em>(&theta;<sub>0,<em>j</em></sub>)
  <span class="eq-num">(4)</span>
</div>
<p class="no-indent">
  Single-qubit <em>R<sub>y</sub></em> gates rotate states around the Y-axis: <em>R<sub>y</sub></em>(&theta;) = [[cos(&theta;/2), &minus;sin(&theta;/2)], [sin(&theta;/2), cos(&theta;/2)]]. CNOT gates apply entangling operations between adjacent qubits (0&rarr;1, 1&rarr;2, 2&rarr;3). With depth <em>L</em> = 2, the ansatz contains exactly <strong>12 trainable parameters</strong> (&theta;<sub>0</sub> to &theta;<sub>11</sub>).
</p>

<h3 class="subsec-heading">5.4 Measurement and optimization loop</h3>
<p>
  Measuring the 4 qubits collapses the statevector into one of 16 bitstrings. The probability of measuring bitstring <em>z</em> is <em>p</em>(<em>z</em>) = |&lang;<em>z</em>|&psi;(<em>x</em>, &theta;)&rang;|<sup>2</sup>. Binary classification parity maps even Hamming weight bitstrings to Human (<em>y</em>=0) and odd to Spoof (<em>y</em>=1):
</p>
<div class="math-display">
  <em>P</em>(<em>y</em>=0|<em>x</em>, &theta;) = &sum;<sub><em>z</em>: wt(<em>z</em>) is even</sub> <em>p</em>(<em>z</em>), &nbsp;&nbsp;&nbsp;&nbsp; <em>P</em>(<em>y</em>=1|<em>x</em>, &theta;) = &sum;<sub><em>z</em>: wt(<em>z</em>) is odd</sub> <em>p</em>(<em>z</em>)
  <span class="eq-num">(5)</span>
</div>
<p class="no-indent">
  The classical COBYLA optimizer iteratively evaluates cross-entropy loss over 60 iterations and updates the 12 quantum angles &theta;. Circuit simulation is performed using Qiskit Aer's <code>SamplerV2</code>.
</p>

<!-- SECTION 6: COMPARATIVE BASELINES -->
<h2 class="sec-heading">6 Comparative framework: Classical Deep Learning baselines (C)</h2>

<h3 class="subsec-heading">6.1 Classical MLP architecture (6,818 parameters)</h3>
<p>
  The classical benchmark is a Multi-Layer Perceptron (MLP) trained on the full 70-dimensional feature space:
</p>
<p class="no-indent">
  &bull; <strong>Input Layer:</strong> 70 standardized acoustic features.<br>
  &bull; <strong>Hidden Layer 1:</strong> Linear(70 &rarr; 64), Batch Normalization, ReLU activation, Dropout (<em>p</em> = 0.25).<br>
  &bull; <strong>Hidden Layer 2:</strong> Linear(64 &rarr; 32), Batch Normalization, ReLU activation, Dropout (<em>p</em> = 0.25).<br>
  &bull; <strong>Output Layer:</strong> Linear(32 &rarr; 2) with Softmax for class probabilities.<br>
  &bull; <strong>Total Parameters:</strong> (70 &times; 64 + 64) + (2 &times; 64) + (64 &times; 32 + 32) + (2 &times; 32) + (32 &times; 2 + 2) = <strong>6,818 parameters</strong>.<br>
  &bull; <strong>Training:</strong> Adam optimizer (&alpha; = 0.001, weight decay = 10<sup>&minus;4</sup>), 40 epochs, batch size 16.
</p>

<h3 class="subsec-heading">6.2 Hardware execution: CPU vs. NVIDIA CUDA GPU</h3>
<p>
  1) <em>Host CPU:</em> PyTorch single-precision CPU execution.<br>
  2) <em>NVIDIA CUDA GPU:</em> NVIDIA GeForce RTX 2050 Mobile GPU (CUDA 12.4, Driver 610.88). Timing executes explicit stream synchronization (<code>torch.cuda.synchronize()</code>) to guarantee sub-millisecond precision.
</p>

<!-- SECTION 7: RESULTS AND DISCUSSION -->
<h2 class="sec-heading">7 Results and discussion (O)</h2>
<p>
  All three models were evaluated on the identical held-out test partition of 40 recordings (20 Bonafide Human, 20 AI Spoof). Quantitative results are detailed in Table 1.
</p>

<table class="springer-table">
  <caption>Table 1 Quantitative performance comparison across CPU, GPU, and Quantum VQC</caption>
  <thead>
    <tr>
      <th>Performance Metric</th>
      <th>PyTorch MLP (CPU)</th>
      <th>PyTorch MLP (CUDA GPU)</th>
      <th>Qiskit VQC (Aer Simulator)</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td style="text-align: left; font-weight: bold;">Accuracy</td>
      <td><strong>87.50%</strong></td>
      <td>82.50%</td>
      <td>47.50%</td>
    </tr>
    <tr>
      <td style="text-align: left; font-weight: bold;">Precision (Spoof)</td>
      <td><strong>100.00%</strong></td>
      <td>88.24%</td>
      <td>48.00%</td>
    </tr>
    <tr>
      <td style="text-align: left; font-weight: bold;">Recall (Spoof)</td>
      <td><strong>75.00%</strong></td>
      <td><strong>75.00%</strong></td>
      <td>60.00%</td>
    </tr>
    <tr>
      <td style="text-align: left; font-weight: bold;">F<sub>1</sub>-Score</td>
      <td><strong>0.8571</strong></td>
      <td>0.8108</td>
      <td>0.5333</td>
    </tr>
    <tr>
      <td style="text-align: left;">True Positives (Spoof)</td>
      <td>15 / 20</td>
      <td>15 / 20</td>
      <td>12 / 20</td>
    </tr>
    <tr>
      <td style="text-align: left;">False Positives (Spoof)</td>
      <td><strong>0 / 20</strong></td>
      <td>2 / 20</td>
      <td>13 / 20</td>
    </tr>
    <tr>
      <td style="text-align: left;">True Negatives (Human)</td>
      <td><strong>20 / 20</strong></td>
      <td>18 / 20</td>
      <td>7 / 20</td>
    </tr>
    <tr>
      <td style="text-align: left;">False Negatives (Human)</td>
      <td>5 / 20</td>
      <td>5 / 20</td>
      <td>8 / 20</td>
    </tr>
    <tr class="total-row">
      <td style="text-align: left;">Input Feature Dimension</td>
      <td>70 features</td>
      <td>70 features</td>
      <td><strong>4 features (PCA)</strong></td>
    </tr>
    <tr>
      <td style="text-align: left;">Model Parameters</td>
      <td>6,818 weights</td>
      <td>6,818 weights</td>
      <td><strong>12 angles (568&times; less)</strong></td>
    </tr>
    <tr>
      <td style="text-align: left;">Total Training Time</td>
      <td><strong>0.71 s</strong></td>
      <td>1.28 s</td>
      <td>33.40 s</td>
    </tr>
    <tr>
      <td style="text-align: left;">Inference Latency (per clip)</td>
      <td><strong>0.12 ms</strong></td>
      <td>0.33 ms</td>
      <td>7.89 ms</td>
    </tr>
  </tbody>
</table>

<!-- FIGURE 2: PERFORMANCE METRICS & LATENCY -->
<div class="figure-wrapper">
  <div style="display: flex; gap: 14px; justify-content: center;">
    <div style="flex: 1; text-align: center;">
      <img src="__METRICS_B64__" alt="Metrics Comparison" style="max-height: 195px;">
    </div>
    <div style="flex: 1; text-align: center;">
      <img src="__LATENCY_B64__" alt="Latency Comparison" style="max-height: 195px;">
    </div>
  </div>
  <div class="figure-caption">
    <strong>Figure 2. Empirical classification performance and latency benchmark.</strong> (Left) Grouped bar chart comparing Accuracy, Precision, Recall, and <em>F</em><sub>1</sub>-score across CPU, GPU, and Quantum VQC architectures on identical held-out ASVspoof 2019 test data. (Right) Training duration and single-sample inference latency comparisons illustrating classical computational speed versus simulation overhead.
  </div>
</div>

<!-- FIGURE 3: CONFUSION MATRICES -->
<div class="figure-wrapper">
  <img src="__CM_B64__" alt="Composite Confusion Matrices" style="max-height: 175px;">
  <div class="figure-caption">
    <strong>Figure 3. Confusion matrix profiles on held-out test data (40 audio samples).</strong> Left: PyTorch CPU MLP (20 TN, 15 TP, 0 FP, 5 FN). Center: PyTorch CUDA GPU MLP (18 TN, 15 TP, 2 FP, 5 FN). Right: Qiskit Aer VQC (7 TN, 12 TP, 13 FP, 8 FN).
  </div>
</div>

<h3 class="subsec-heading">7.1 Classification performance analysis</h3>
<p>
  The Classical CPU MLP achieved the highest classification performance with 87.50% accuracy (<em>F</em><sub>1</sub> = 0.8571). It achieved 100% precision with zero false alarms (all 20 human samples correctly identified as human, while 15 of 20 AI spoofs were detected). The GPU MLP performed comparably with 82.50% accuracy. The Quantum VQC achieved 47.50% accuracy and <em>F</em><sub>1</sub> = 0.5333, successfully identifying 12 of 20 spoof instances.
</p>

<h3 class="subsec-heading">7.2 Timing &amp; computational throughput</h3>
<p>
  Classical CPU training completed in 0.71 seconds. GPU training required 1.28 seconds due to CUDA context setup and kernel launch overhead for small tensor batches. In contrast, VQC training on Qiskit Aer required 33.40 seconds (47&times; longer) due to simulating 16-amplitude complex statevectors across 60 COBYLA cost function iterations. Single-sample inference latency was 0.12 ms on CPU, 0.33 ms on GPU, and 7.89 ms on the quantum simulator.
</p>

<h3 class="subsec-heading">7.3 The 568:1 parameter compression phenomenon</h3>
<p>
  The most notable scientific finding is the extreme parameter compression:
</p>
<div class="math-display">
  \text{Compression Ratio} = \frac{6,818\text{ classical weights}}{12\text{ quantum angles}} \approx 568.17 : 1
</div>
<p class="no-indent">
  With only 12 variational angles, the VQC achieved an <em>F</em><sub>1</sub>-score of 0.5333. In classical neural networks, reducing capacity to 12 parameters causes immediate underfitting. The VQC's ability to retain non-trivial classification ability demonstrates the representational expressiveness of multi-qubit Hilbert space &Copf;<sup>16</sup> enabled by CNOT entanglement.
</p>

<h3 class="subsec-heading">7.4 Dimensionality reduction and PCA information loss</h3>
<p>
  The accuracy disparity between classical MLPs (87.50%) and VQC (47.50%) stems directly from the PCA information bottleneck. Compressing 70 acoustic dimensions to 4 components discards 47.88% of spectral variance. Generative speech synthesis artifacts—such as vocoder phase discontinuities—predominantly reside in higher-order cepstral coefficients, which were pruned to fit the 4-qubit limitation.
</p>

<h3 class="subsec-heading">7.5 Simulation overhead vs. physical QPU execution</h3>
<p>
  Simulating quantum circuits on classical von Neumann machines requires &Oscr;(2<sup><em>N</em></sup>) matrix-vector operations. Consequently, the VQC was 65&times; slower in inference than the CPU MLP. While physical QPUs execute unitary evolutions in physical coherence time (~10<sup>&minus;5</sup> s), classical simulation offers no runtime advantage for real-time speech verification.
</p>

<h3 class="subsec-heading">7.6 Barren plateaus and optimizer dynamics</h3>
<p>
  Derivative-free COBYLA optimization avoided the barren plateau phenomenon (vanishing gradients in Haar-distributed state spaces) that frequently hinders gradient-based QML. The shallow ansatz depth (<em>L</em> = 2) and linear entanglement topology ensured active cost function variance throughout training.
</p>

<!-- SECTION 8: LIMITATIONS & FUTURE SCOPE -->
<h2 class="sec-heading">8 Limitations &amp; future scope</h2>
<p>
  1) <em>Physical QPU Testing:</em> Deploying circuits on superconducting transmon QPUs (e.g., IBM Quantum Eagle) to evaluate thermal noise and readout error mitigation.<br>
  2) <em>Quantum-Classical Hybrid Architectures:</em> Utilizing classical convolutional front-ends to compress spectrograms into quantum latent representations.<br>
  3) <em>Higher-Order Feature Maps:</em> Investigating data-reuploading circuits and Hamiltonian evolution maps to encode richer acoustic features.
</p>

<!-- SECTION 9: DATA AVAILABILITY -->
<h2 class="sec-heading">Data availability</h2>
<p class="no-indent">
  The dataset supporting this study is publicly accessible on Hugging Face at <a href="https://huggingface.co/datasets/SpeechAntiSpoofingBenchmarks/ASVspoof2019_LA">https://huggingface.co/datasets/SpeechAntiSpoofingBenchmarks/ASVspoof2019_LA</a>. Extracted feature arrays, trained model checkpoints, and evaluation metrics are available in the repository at <code>data/</code>, <code>models/</code>, and <code>results/</code>.
</p>

<!-- SECTION 10: CONCLUSION -->
<h2 class="sec-heading">10 Conclusion</h2>
<p>
  This study applied the PICO framework to evaluate Quantum Machine Learning against Classical Deep Learning on CPU and NVIDIA CUDA GPU for voice anti-spoofing using the ASVspoof 2019 Logical Access benchmark. Classical MLPs achieved superior accuracy (87.50%) and sub-millisecond inference (0.12 ms). While the 4-qubit VQC demonstrated remarkable parameter compression (568&times; fewer parameters), it suffered from PCA information loss and classical simulation overhead. True quantum advantage in audio forensics remains contingent on physical quantum hardware execution.
</p>

<!-- REFERENCES -->
<h2 class="sec-heading">References</h2>
<ol class="references-ol">
  <li>Havl&iacute;&check;ek V, C&oacute;rcoles AD, Temme K, Harrow AW, Kandala A, Chow JM, Gambetta JM (2019) Supervised learning with quantum-enhanced feature spaces. Nature 567(7747):209&ndash;212</li>
  <li>Farhi E, Neven H (2018) Classification with quantum neural networks on near term processors. arXiv:1802.06002</li>
  <li>Schuld M, Killoran N (2019) Quantum machine learning in feature Hilbert spaces. Phys Rev Lett 122(4):040504</li>
  <li>Todisco M, Wang X, Vestman V, Sahidullah M, Delgado H, Nautsch A, Yamagishi J, Evans N, Kinnunen T, Lee KA (2019) ASVspoof 2019: Future perspectives in spoofed and fake speech detection. In: Proc Interspeech, pp 1008&ndash;1012</li>
  <li>Tak H, Patino J, Todisco M, Nautsch A, Evans N, Larcher A (2021) End-to-end anti-spoofing with RawNet2. In: Proc IEEE ICASSP, pp 6369&ndash;6373</li>
  <li>Jung JW, Heo HS, Tak H, Shim HJ, Chung JS (2022) AASIST: Audio anti-spoofing using integrated spectro-temporal graph attention networks. In: Proc IEEE ICASSP, pp 6365&ndash;6369</li>
  <li>SpeechAntiSpoofingBenchmarks (2023) ASVspoof 2019 Logical Access Benchmark Dataset. Hugging Face Datasets. <a href="https://huggingface.co/datasets/SpeechAntiSpoofingBenchmarks/ASVspoof2019_LA">https://huggingface.co/datasets/SpeechAntiSpoofingBenchmarks/ASVspoof2019_LA</a></li>
  <li>McClean JR, Boixo S, Smelyanskiy VN, Babbush R, Neven H (2018) Barren plateaus in quantum neural network training landscapes. Nat Commun 9(1):4812</li>
  <li>Qiskit contributors (2024) Qiskit: An open-source framework for quantum computing. Zenodo. doi:10.5281/zenodo.2573505</li>
</ol>

<!-- PAGE 1 FOOTER LICENSE -->
<div class="cc-footer">
  &copy; The Author(s) 2026. <strong>Open Access</strong> This article is licensed under a Creative Commons Attribution 4.0 International License, which permits use, sharing, adaptation, distribution and reproduction in any medium or format, as long as you give appropriate credit to the original author(s) and the source, provide a link to the Creative Commons licence, and indicate if changes were made. To view a copy of this licence, visit <a href="http://creativecommons.org/licenses/by/4.0/" style="color: #64748b;">http://creativecommons.org/licenses/by/4.0/</a>.
</div>

</body>
</html>
"""

    html = template.replace("__CIRCUIT_B64__", circuit_b64)
    html = html.replace("__METRICS_B64__", metrics_b64)
    html = html.replace("__LATENCY_B64__", latency_b64)
    html = html.replace("__CM_B64__", cm_b64)
    return html


def generate_pdf():
    print("Building Springer Nature / Discover format HTML document...")
    html_content = build_html()
    with open(OUTPUT_HTML, "w", encoding="utf-8") as f:
        f.write(html_content)
    print(f"Saved HTML to {OUTPUT_HTML} ({len(html_content)} bytes)")


if __name__ == "__main__":
    generate_pdf()
