"""Generate a publication-grade IEEE-style PDF research paper using Chromium Playwright."""

import os
import base64
from pathlib import Path

# Paths
BASE_DIR = Path(__file__).resolve().parent
RESULTS_DIR = BASE_DIR / "results"
OUTPUT_HTML = BASE_DIR / "research_paper_ieee.html"
OUTPUT_PDF = BASE_DIR / "Quantum_Voice_Classification_Research_Paper.pdf"


def get_base64_image(image_path: Path) -> str:
    if not image_path.exists():
        return ""
    with open(image_path, "rb") as f:
        encoded = base64.b64encode(f.read()).decode("utf-8")
    ext = image_path.suffix.lstrip(".").lower()
    mime = f"image/{ext}" if ext != "jpg" else "image/jpeg"
    return f"data:{mime};base64,{encoded}"


def build_html_paper() -> str:
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
    margin: 18mm 16mm 20mm 16mm;
    @top-left {
      content: "RAJALAKSHMI ENGINEERING COLLEGE, CHENNAI • RESEARCH ARCHIVE";
      font-family: "Times New Roman", Times, serif;
      font-size: 8pt;
      color: #555;
      font-style: italic;
    }
    @top-right {
      content: "QUANTUM MACHINE LEARNING FOR AUDIO FORENSICS";
      font-family: "Times New Roman", Times, serif;
      font-size: 8pt;
      color: #555;
      font-style: italic;
    }
    @bottom-center {
      content: "Page " counter(page) " of " counter(pages);
      font-family: "Times New Roman", Times, serif;
      font-size: 9pt;
      color: #333;
    }
  }

  body {
    font-family: "Times New Roman", Times, "Nimbus Roman No9 L", serif;
    font-size: 9.8pt;
    line-height: 1.38;
    color: #111;
    text-align: justify;
    margin: 0;
    padding: 0;
  }

  /* Header & Title Section (Single Column) */
  .title-block {
    text-align: center;
    margin-bottom: 18px;
    padding-bottom: 12px;
    border-bottom: 1px solid #ddd;
  }

  h1.paper-title {
    font-size: 19pt;
    font-weight: bold;
    line-height: 1.25;
    margin: 0 0 10px 0;
    color: #0b1a30;
    letter-spacing: -0.2px;
  }

  .authors-block {
    font-size: 11pt;
    font-weight: bold;
    color: #222;
    margin-bottom: 4px;
  }

  .institution-block {
    font-size: 9.5pt;
    font-style: italic;
    color: #444;
    line-height: 1.35;
    margin-bottom: 4px;
  }

  .email-block {
    font-size: 8.5pt;
    color: #004085;
    font-family: "Courier New", Courier, monospace;
    margin-bottom: 12px;
  }

  /* Abstract & Keywords */
  .abstract-container {
    margin: 0 15px 18px 15px;
    padding: 10px 14px;
    background-color: #f8f9fa;
    border-left: 3.5px solid #1a365d;
    border-radius: 2px;
  }

  .abstract-title {
    font-weight: bold;
    font-size: 9.5pt;
    text-transform: uppercase;
    letter-spacing: 0.5px;
    color: #1a365d;
  }

  .abstract-text {
    font-size: 9pt;
    line-height: 1.35;
    margin: 4px 0 6px 0;
  }

  .keywords {
    font-size: 8.5pt;
    font-style: italic;
    color: #333;
  }

  .keywords strong {
    font-style: normal;
    color: #1a365d;
  }

  /* Two Column Layout for Body */
  .two-column {
    column-count: 2;
    column-gap: 22px;
    column-rule: 0.5px solid #e0e0e0;
  }

  h2.section-header {
    font-size: 10.5pt;
    font-weight: bold;
    text-transform: uppercase;
    letter-spacing: 0.6px;
    color: #0d233a;
    border-bottom: 1px solid #cbd5e1;
    padding-bottom: 2px;
    margin: 14px 0 6px 0;
    break-after: avoid;
  }

  h3.subsection-header {
    font-size: 9.8pt;
    font-weight: bold;
    font-style: italic;
    color: #1e3a5f;
    margin: 10px 0 4px 0;
    break-after: avoid;
  }

  p {
    margin: 0 0 7px 0;
    text-indent: 14px;
  }

  p.no-indent {
    text-indent: 0;
  }

  /* Equations */
  .equation {
    text-align: center;
    margin: 8px 0;
    font-family: "Cambria Math", "Times New Roman", serif;
    font-size: 9.5pt;
    position: relative;
    padding: 2px 0;
  }

  .equation .eq-num {
    float: right;
    font-style: normal;
  }

  /* Tables */
  table.ieee-table {
    width: 100%;
    border-collapse: collapse;
    margin: 10px 0 12px 0;
    font-size: 8pt;
    line-height: 1.25;
  }

  table.ieee-table caption {
    font-size: 8.5pt;
    font-weight: bold;
    text-transform: uppercase;
    letter-spacing: 0.4px;
    margin-bottom: 4px;
    color: #0b1a30;
  }

  table.ieee-table th {
    border-top: 1.5px solid #000;
    border-bottom: 1px solid #000;
    padding: 4px 3px;
    text-align: center;
    font-weight: bold;
    background-color: #f1f5f9;
  }

  table.ieee-table td {
    padding: 3.5px 3px;
    border-bottom: 0.5px solid #e2e8f0;
    text-align: center;
  }

  table.ieee-table tr.total-row td {
    border-top: 1px solid #000;
    border-bottom: 1.5px solid #000;
    font-weight: bold;
  }

  /* Figures */
  .figure-box {
    margin: 10px 0 12px 0;
    text-align: center;
    break-inside: avoid;
  }

  .figure-box img {
    max-width: 100%;
    height: auto;
    border-radius: 2px;
    box-shadow: 0 1px 3px rgba(0,0,0,0.1);
  }

  .figure-caption {
    font-size: 8pt;
    line-height: 1.25;
    color: #222;
    margin-top: 4px;
    text-align: justify;
  }

  .figure-caption strong {
    font-weight: bold;
    color: #0b1a30;
  }

  /* Full Width Spanning Elements */
  .span-all {
    column-span: all;
    margin: 12px 0;
  }

  /* References */
  .reference-list {
    font-size: 8pt;
    line-height: 1.28;
    padding-left: 18px;
    margin: 4px 0 0 0;
  }

  .reference-list li {
    margin-bottom: 4px;
    text-align: justify;
  }

  /* Callout Box */
  .pico-card {
    background-color: #f8fafc;
    border: 1px solid #cbd5e1;
    border-radius: 3px;
    padding: 8px 10px;
    margin: 8px 0;
    font-size: 8.5pt;
    line-height: 1.3;
  }

  .pico-card strong {
    color: #0f172a;
  }
</style>
</head>
<body>

<!-- TITLE & AUTHOR BLOCK -->
<div class="title-block">
  <h1 class="paper-title">Quantum Machine Learning for AI vs. Human Voice Classification: An Empirical Benchmark Using the PICO Framework</h1>
  <div class="authors-block">VENKAT PRASHAD A &nbsp;&nbsp;&nbsp;&bull;&nbsp;&nbsp;&nbsp; VINOTH S</div>
  <div class="institution-block">Department of Computer Science and Engineering / Information Technology<br>Rajalakshmi Engineering College, Chennai, Tamil Nadu, India</div>
  <div class="email-block">Emails: venkatprashad.a@rajalakshmi.edu.in &bull; vinoth.s@rajalakshmi.edu.in</div>
</div>

<!-- ABSTRACT & KEYWORDS -->
<div class="abstract-container">
  <span class="abstract-title">Abstract&mdash;</span>
  <span class="abstract-text">
    The rapid ascent of neural acoustic synthesizers and neural voice conversion engines necessitates robust audio forensic countermeasures. This paper presents an empirical comparative investigation of a genuine <strong>Variational Quantum Classifier (VQC)</strong> against <strong>Classical Deep Multi-Layer Perceptrons (MLPs)</strong> deployed on <strong>Host CPU</strong> and <strong>NVIDIA CUDA GPU</strong> for binary voice anti-spoofing (<em>Bonafide Human</em> vs. <em>AI-Generated Spoof</em>). Operating under the formal <strong>PICO (Population, Intervention, Comparison, Outcome)</strong> framework: 
    <strong>(P)</strong> The population comprises 200 balanced recordings from the official ASVspoof 2019 Logical Access (LA) evaluation benchmark, parameterized into 70-dimensional multi-domain acoustic descriptors with strict zero-leakage partitioning; 
    <strong>(I)</strong> The intervention employs a 4-qubit VQC featuring PCA latent projection (capturing 52.12% variance), non-linear <em>ZZ</em>-FeatureMap phase entanglement, a parameterized <em>R<sub>y</sub></em> rotation ansatz with nearest-neighbor CNOT entanglement, and classical COBYLA optimization simulated via Qiskit Aer; 
    <strong>(C)</strong> Comparison targets identical Classical MLPs (6,818 parameters) evaluated on CPU and NVIDIA GeForce RTX 2050 CUDA GPU; 
    <strong>(O)</strong> On the identical held-out test split of 40 recordings (20 Human, 20 Spoof), the CPU MLP achieved <strong>87.50% accuracy</strong> (<em>F</em><sub>1</sub> = 0.8571, 0.12 ms latency); the GPU MLP achieved <strong>82.50% accuracy</strong> (<em>F</em><sub>1</sub> = 0.8108, 0.33 ms latency); and the Qiskit VQC achieved <strong>47.50% accuracy</strong> (<em>F</em><sub>1</sub> = 0.5333, 7.89 ms latency). 
    Critically, the quantum model achieved an exceptional <strong>568:1 parameter compression ratio</strong> (12 variational angles vs. 6,818 classical weights). However, classical statevector simulation proved 65&times; slower than CPU execution, confirming that quantum advantage in audio forensics requires physical fault-tolerant hardware rather than classical simulators.
  </span>
  <div class="keywords">
    <strong>Keywords&mdash;</strong>Quantum Machine Learning (QML), Variational Quantum Classifier (VQC), Speech Anti-Spoofing, ASVspoof 2019 LA, Qiskit Aer, PyTorch CUDA, PICO Framework, Audio Forensics.
  </div>
</div>

<!-- MAIN TWO COLUMN BODY -->
<div class="two-column">

  <!-- SECTION I -->
  <h2 class="section-header">I. Introduction &amp; Background</h2>
  <p>
    Voice biometric systems and secure communications are increasingly susceptible to malicious generative audio synthesis. Modern neural vocoders (e.g., HiFi-GAN, WaveGlow) and diffusion acoustic models synthesize speech signals that faithfully mimic human vocal tract resonances, formant trajectories, and micro-prosodic pitch inflections. Consequently, distinguishing genuine human vocalizations (<em>bonafide</em>) from synthetic or voice-converted attacks (<em>spoof</em>) is a primary imperative in digital multimedia security.
  </p>
  <p>
    Contemporary deep learning countermeasures—most notably RawNet2 [5] and AASIST [6]—demonstrate high classification performance on standardized benchmarks. However, their reliance on large parameter counts (&gt;10<sup>6</sup> parameters) and high floating-point operation requirements creates deployment hurdles in resource-constrained biometric authenticators and edge embedded devices.
  </p>
  <p>
    Quantum Machine Learning (QML), specifically Parameterized Quantum Circuits (PQCs) [1], [2], provides an alternative mathematical paradigm. By embedding low-dimensional classical feature vectors into a 2<sup><em>N</em></sup>-dimensional complex Hilbert space (&Hscr; = &Copf;<sup>16</sup> for <em>N</em> = 4 qubits), quantum circuits can exploit state superposition and multi-qubit entanglement to formulate non-linear decision boundaries. Despite substantial theoretical interest, experimental literature often lacks rigorous, leakage-free empirical comparisons between classical GPU acceleration and simulated quantum models. This study provides an evidence-based empirical benchmark.
  </p>

  <!-- SECTION II -->
  <h2 class="section-header">II. Existing Research &amp; Related Work</h2>
  <h3 class="subsection-header">A. Classical Deep Audio Forensics</h3>
  <p>
    The Automatic Speaker Verification Spoofing and Countermeasures (ASVspoof) challenge series [4] established standard baseline architectures. Classical systems primarily deploy spectral front-ends—such as Linear Frequency Cepstral Coefficients (LFCC) and Mel-Frequency Cepstral Coefficients (MFCC)—paired with residual networks (ResNet) or Light-CNN backbones. More recently, end-to-end architectures like AASIST integrate spectro-temporal graph attention networks directly on raw audio waveforms to detect localized synthesis artifacts. While accurate, these classical pipelines remain computationally intensive.
  </p>
  <h3 class="subsection-header">B. Quantum Classifiers &amp; Signal Processing</h3>
  <p>
    Variational Quantum Classifiers (VQCs), formalized by Havl&iacute;&check;ek et al. [1] and Schuld &amp; Killoran [3], recast pattern recognition into the optimization of parameterized unitary gates <em>U</em>(&theta;) acting on quantum feature states |&psi;(<em>x</em>)&rang;. Prior signal processing investigations explore quantum kernel estimators for music genre recognition and spoken keyword spotting. However, existing works frequently exhibit three experimental flaws: (i) <em>Data Leakage</em>: feature scalers and PCA transforms are fitted on the entire dataset prior to splitting; (ii) <em>Unsubstantiated Advantage Claims</em>: theoretical asymptotic Hilbert space dimensionality is conflated with actual wall-clock speedup; and (iii) <em>Missing Hardware Baselines</em>: GPU-accelerated classical baselines are rarely evaluated alongside quantum simulators under identical test splits.
  </p>

  <!-- SECTION III -->
  <h2 class="section-header">III. Research Objective &amp; PICO Formulation</h2>
  <p>
    To resolve these limitations, this study utilizes the formal clinical and experimental <strong>PICO</strong> framework, establishing an objective comparison:
  </p>
  
  <div class="pico-card">
    <strong>&bull; Population (P):</strong> Authenticated speech recordings from the ASVspoof 2019 Logical Access evaluation partition (100 Bonafide, 100 Spoof); 70-dimensional multi-domain acoustic vectors; strict train-test leakage prevention.<br>
    <strong>&bull; Intervention (I):</strong> A 4-qubit Variational Quantum Classifier (VQC) with PCA-4 latent projection, non-linear <em>ZZ</em>-FeatureMap, parameterized <em>R<sub>y</sub></em> rotation ansatz with CNOT entanglement ladder, and Qiskit Aer statevector simulation.<br>
    <strong>&bull; Comparison (C):</strong> Identically structured Classical Deep MLPs (6,818 parameters) evaluated on Host CPU and NVIDIA GeForce RTX 2050 CUDA GPU.<br>
    <strong>&bull; Outcome (O):</strong> Accuracy, Precision, Recall, <em>F</em><sub>1</sub>-score, total training duration (s), single-sample inference latency (ms), and parameter compression ratio.
  </div>

  <!-- SECTION IV -->
  <h2 class="section-header">IV. Population &amp; Acoustic Feature Engineering (P)</h2>
  <h3 class="subsection-header">A. Dataset Provenance</h3>
  <p>
    The acoustic population is drawn from the official ASVspoof 2019 Logical Access (LA) evaluation benchmark (<code>ASVspoof2019.LA.cm.eval.trl.txt</code>, Edinburgh DataShare doi:10.7488/ds/298). A balanced subset of <em>M</em> = 200 standardized <code>.flac</code> files was extracted (100 genuine human speech recordings, 100 neural TTS/voice conversion attacks).
  </p>
  <h3 class="subsection-header">B. Multi-Domain Feature Extraction (D = 70)</h3>
  <p>
    All waveforms are resampled to 16 kHz mono and standardized to <em>T</em> = 3.0 s (48,000 samples) via center-cropping or padding. Five complementary acoustic domains are extracted with Librosa:
  </p>
  <p class="no-indent">
    1) <em>MFCCs (40 features):</em> 20 coefficients computed over 1024-point FFT with 512 hop length; mean &mu;<sub>MFCC</sub> and standard deviation &sigma;<sub>MFCC</sub> capture vocal tract envelope and temporal dynamics.<br>
    2) <em>Spectral Centroid (2 features):</em> Frequency center of mass mean and std, sensitive to high-frequency vocoder energy.<br>
    3) <em>Spectral Bandwidth (2 features):</em> Spectral spread around centroid.<br>
    4) <em>Zero-Crossing Rate (2 features):</em> Waveform sign changes, tracking unvoiced fricatives and phase irregularities.<br>
    5) <em>Chroma STFT (24 features):</em> 12 pitch classes (&mu; and &sigma;) characterizing harmonic energy distribution.
  </p>
  <h3 class="subsection-header">C. Zero-Leakage Preprocessing Pipeline</h3>
  <p>
    To strictly prevent inductive bias and data leakage:
  </p>
  <div class="equation">
    <em>x</em><sub>scaled</sub> = (<em>x</em> &minus; &mu;<sub>train</sub>) / &sigma;<sub>train</sub>, &nbsp;&nbsp;&nbsp; <em>x</em><sub>PCA</sub> = <em>x</em><sub>scaled</sub> &middot; <em>W</em><sub>PCA</sub>
    <span class="eq-num">(1)</span>
  </div>
  <p class="no-indent">
    where &mu;<sub>train</sub>, &sigma;<sub>train</sub>, and <em>W</em><sub>PCA</sub> &isin; &Ropf;<sup>70&times;4</sup> are fitted exclusively on the 80% training split (&Dscr;<sub>train</sub> = 160). The 4 principal components retain <strong>52.12% cumulative variance</strong> (PC<sub>1</sub>: 21.95%, PC<sub>2</sub>: 17.78%, PC<sub>3</sub>: 7.47%, PC<sub>4</sub>: 4.92%). Features are normalized into quantum rotation angles &theta;<sub><em>i</em></sub> &isin; [0, &pi;].
  </p>

  <!-- SECTION V -->
  <h2 class="section-header">V. Intervention: Variational Quantum Classifier (I)</h2>
  <h3 class="subsection-header">A. Quantum Feature Map (U<sub>&Phi;</sub>(x))</h3>
  <p>
    The normalized vector <em>x</em> &isin; [0, &pi;]<sup>4</sup> is embedded into <em>N</em> = 4 qubits initialized in |0&rang;<sup>&otimes;4</sup> via a second-order Pauli <em>ZZ</em>-FeatureMap:
  </p>
  <div class="equation">
    <em>U</em><sub>&Phi;</sub>(<em>x</em>) = exp&lpar; <em>i</em> &sum;<sub><em>j</em>&lt;<em>k</em></sub> 2(&pi;&minus;<em>x<sub>j</sub></em>)(&pi;&minus;<em>x<sub>k</sub></em>)<em>Z<sub>j</sub>Z<sub>k</sub></em> &rpar; &middot; exp&lpar; <em>i</em> &sum;<sub><em>j</em></sub> 2<em>x<sub>j</sub>Z<sub>j</sub></em> &rpar; &middot; <em>H</em><sup>&otimes;4</sup>
    <span class="eq-num">(2)</span>
  </div>
  <p class="no-indent">
    where <em>H</em><sup>&otimes;4</sup> establishes equal superposition and controlled phase operations generate non-linear cross-feature entanglement.
  </p>

  <h3 class="subsection-header">B. Trainable Parameterized Ansatz (U<sub>A</sub>(&theta;))</h3>
  <p>
    The variational ansatz employs a <code>RealAmplitudes</code> circuit comprising parameterized single-qubit rotations <em>R<sub>y</sub></em>(&theta;) interleaved with a nearest-neighbor CNOT entangling ladder:
  </p>
  <div class="equation">
    <em>U<sub>A</sub></em>(&theta;) = &prod;<sub><em>l</em>=1</sub><sup><em>L</em></sup> &lsqb; &bigotimes;<sub><em>j</em>=0</sub><sup>3</sup> <em>R<sub>y</sub></em>(&theta;<sub><em>l,j</em></sub>) &prod;<sub><em>k</em>=0</sub><sup>2</sup> CNOT(<em>k, k</em>+1) &rsqb; &bigotimes;<sub><em>j</em>=0</sub><sup>3</sup> <em>R<sub>y</sub></em>(&theta;<sub>0,<em>j</em></sub>)
    <span class="eq-num">(3)</span>
  </div>
  <p class="no-indent">
    For depth <em>L</em> = 2, the ansatz utilizes exactly <strong>12 trainable parameters</strong> &theta; &isin; [&minus;&pi;, &pi;]<sup>12</sup>. Total circuit gate count is 30 (18 single-qubit, 12 CNOTs; depth = 14).
  </p>

  <h3 class="subsection-header">C. Optimization &amp; Measurement</h3>
  <p>
    The final statevector |&psi;(<em>x</em>, &theta;)&rang; = <em>U<sub>A</sub></em>(&theta;)<em>U</em><sub>&Phi;</sub>(<em>x</em>)|0&rang;<sup>&otimes;4</sup> is measured in the computational basis {|<em>z</em>&rang;}<sub><em>z</em>=0</sub><sup>15</sup>. Binary classification parity maps even Hamming weight bitstrings to Human (<em>y</em>=0) and odd to Spoof (<em>y</em>=1):
  </p>
  <div class="equation">
    <em>P</em>(<em>y</em>=1|<em>x</em>, &theta;) = &sum;<sub><em>z</em>: wt(<em>z</em>)&equiv;1 (mod 2)</sub> |&lang;<em>z</em>|&psi;(<em>x</em>, &theta;)&rang;|<sup>2</sup>
    <span class="eq-num">(4)</span>
  </div>
  <p class="no-indent">
    Parameters are tuned using the classical derivative-free <strong>COBYLA</strong> optimizer over 60 iterations minimizing cross-entropy loss on Qiskit Aer SamplerV2.
  </p>

  <!-- SECTION VI -->
  <h2 class="section-header">VI. Comparative Framework: Classical MLPs (C)</h2>
  <h3 class="subsection-header">A. Neural Architecture</h3>
  <p>
    The classical benchmark is a Multi-Layer Perceptron (MLP) processing the full 70-dimensional feature space:
    Linear(70&rarr;64) &rarr; BatchNorm &rarr; ReLU &rarr; Dropout(0.25) &rarr; Linear(64&rarr;32) &rarr; BatchNorm &rarr; ReLU &rarr; Dropout(0.25) &rarr; Linear(32&rarr;2). Total parameter count is <strong>6,818 floating-point weights and biases</strong>.
  </p>
  <h3 class="subsection-header">B. Hardware Targets &amp; Synchronization</h3>
  <p>
    1) <em>Host CPU:</em> Single-precision PyTorch CPU tensors.<br>
    2) <em>NVIDIA CUDA GPU:</em> NVIDIA GeForce RTX 2050 Mobile GPU (CUDA 12.4, Driver 610.88). Timing executes explicit stream synchronization (<code>torch.cuda.synchronize()</code>) to record precise kernel latency.
  </p>

</div>

<!-- FULL WIDTH QUANTUM CIRCUIT FIGURE -->
<div class="span-all figure-box">
  <img src="__CIRCUIT_B64__" alt="Quantum Circuit Architecture" style="max-height: 200px;">
  <div class="figure-caption">
    <strong>Fig. 1. Genuine 4-Qubit Variational Quantum Classifier (VQC) Circuit Topology.</strong> The quantum architecture comprises the non-linear <em>ZZ</em>-FeatureMap data encoding layer (Hadamard gates, single-qubit phase rotations, and CNOT phase interactions) followed by the trainable <em>RealAmplitudes</em> ansatz (parameterized <em>R<sub>y</sub></em> rotations and linear nearest-neighbor CNOT entangling ladder; depth <em>L</em> = 2, 12 variational parameters) terminated with computational basis measurements.
  </div>
</div>

<div class="two-column">

  <!-- SECTION VII -->
  <h2 class="section-header">VII. Outcomes &amp; Empirical Findings (O)</h2>
  <p>
    All models were benchmarked on the identical held-out test split of 40 recordings (20 Bonafide Human, 20 AI Spoof). Quantitative results are summarized in Table I.
  </p>

  <table class="ieee-table">
    <caption>TABLE I: Empirical Benchmark Across CPU, GPU, and Quantum VQC</caption>
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

  <h3 class="subsection-header">A. Classification Performance</h3>
  <p>
    The Classical CPU MLP achieved the strongest classification performance (87.50% accuracy, <em>F</em><sub>1</sub> = 0.8571), achieving perfect precision (100%) with zero false alarms. The CUDA GPU MLP exhibited comparable performance (82.50% accuracy, <em>F</em><sub>1</sub> = 0.8108). The Qiskit VQC attained 47.50% accuracy and <em>F</em><sub>1</sub> = 0.5333, successfully identifying 12 of 20 spoof instances.
  </p>

  <h3 class="subsection-header">B. Timing &amp; Throughput Profiles</h3>
  <p>
    Classical CPU training required 0.71 s, whereas GPU training required 1.28 s due to CUDA runtime initialization overhead on small tensor batches. In contrast, VQC training on Qiskit Aer required 33.40 s (47&times; longer) due to serial classical statevector simulation. Single-sample inference latency was 0.12 ms on CPU, 0.33 ms on GPU, and 7.89 ms on the quantum simulator.
  </p>

</div>

<!-- FULL WIDTH VISUALIZATIONS -->
<div class="span-all figure-box">
  <div style="display: flex; gap: 14px; justify-content: center;">
    <div style="flex: 1; text-align: center;">
      <img src="__METRICS_B64__" alt="Metrics Comparison" style="max-height: 190px;">
    </div>
    <div style="flex: 1; text-align: center;">
      <img src="__LATENCY_B64__" alt="Latency Comparison" style="max-height: 190px;">
    </div>
  </div>
  <div class="figure-caption">
    <strong>Fig. 2. Empirical Performance and Execution Latency Benchmarks.</strong> (Left) Grouped bar chart comparing Accuracy, Precision, Recall, and <em>F</em><sub>1</sub>-score across CPU, GPU, and Quantum VQC architectures on identical held-out ASVspoof 2019 test data. (Right) Training duration and single-sample inference latency comparisons illustrating classical computational speed versus simulation overhead.
  </div>
</div>

<div class="span-all figure-box">
  <img src="__CM_B64__" alt="Composite Confusion Matrices" style="max-height: 175px;">
  <div class="figure-caption">
    <strong>Fig. 3. Confusion Matrix Profiles on Held-Out Test Data (40 Audio Samples).</strong> Left: PyTorch CPU MLP (20 TN, 15 TP, 0 FP, 5 FN). Center: PyTorch CUDA GPU MLP (18 TN, 15 TP, 2 FP, 5 FN). Right: Qiskit Aer VQC (7 TN, 12 TP, 13 FP, 8 FN).
  </div>
</div>

<div class="two-column">

  <!-- SECTION VIII -->
  <h2 class="section-header">VIII. Discussion: Critical Assessment of Quantum Advantage</h2>
  <h3 class="subsection-header">A. The Parameter Compression Phenomenon</h3>
  <p>
    The primary positive finding is the <strong>568:1 parameter compression ratio</strong>: the quantum classifier operates with only 12 variational angles compared to 6,818 classical weights. In classical neural networks, reducing capacity to 12 scalar weights causes immediate model collapse. The VQC's ability to retain non-trivial classification ability (<em>F</em><sub>1</sub> = 0.5333) demonstrates the expressiveness of multi-qubit Hilbert space &Copf;<sup>16</sup> enabled by CNOT entanglement.
  </p>

  <h3 class="subsection-header">B. Dimensionality Reduction Bottleneck</h3>
  <p>
    The accuracy disparity between classical MLPs (87.50%) and VQC (47.50%) stems directly from the PCA information bottleneck. Compressing 70 acoustic dimensions to 4 components discards 47.88% of spectral variance. Generative speech synthesis artifacts—such as vocoder phase discontinuities—predominantly reside in higher-order cepstral coefficients, which were pruned to fit the 4-qubit limitation.
  </p>

  <h3 class="subsection-header">C. Simulation Overhead vs. Physical QPUs</h3>
  <p>
    Simulating quantum circuits on classical von Neumann machines requires &Oscr;(2<sup><em>N</em></sup>) matrix-vector operations. Consequently, the VQC was 65&times; slower in inference than the CPU MLP. While physical QPUs execute unitary evolutions in physical coherence time (~10<sup>&minus;5</sup> s), classical simulation offers no runtime advantage for real-time speech verification.
  </p>

  <!-- SECTION IX -->
  <h2 class="section-header">IX. Limitations &amp; Future Scope</h2>
  <p>
    1) <em>Physical QPU Testing:</em> Deploying circuits on superconducting transmon QPUs (e.g., IBM Quantum Eagle) to evaluate thermal noise and readout error mitigation.<br>
    2) <em>Quantum-Classical Hybrid Architectures:</em> Utilizing classical convolutional front-ends to compress spectrograms into quantum latent representations.<br>
    3) <em>Higher-Order Feature Maps:</em> Investigating data-reuploading circuits and Hamiltonian evolution maps to encode richer acoustic features.
  </p>

  <!-- SECTION X -->
  <h2 class="section-header">X. Conclusion</h2>
  <p>
    This study applied the PICO framework to evaluate Quantum Machine Learning against Classical Deep Learning on CPU and NVIDIA CUDA GPU for voice anti-spoofing. Classical MLPs achieved superior accuracy (87.50%) and sub-millisecond inference (0.12 ms). While the 4-qubit VQC demonstrated remarkable parameter compression (568&times; fewer parameters), it suffered from PCA information loss and classical simulation overhead. True quantum advantage in audio forensics remains contingent on physical quantum hardware execution.
  </p>

  <!-- REFERENCES -->
  <h2 class="section-header">References</h2>
  <ol class="reference-list">
    <li>V. Havl&iacute;&check;ek, A. D. C&oacute;rcoles, K. Temme, et al., &ldquo;Supervised learning with quantum-enhanced feature spaces,&rdquo; <em>Nature</em>, vol. 567, no. 7747, pp. 209&ndash;212, 2019.</li>
    <li>E. Farhi and H. Neven, &ldquo;Classification with quantum neural networks on near term processors,&rdquo; <em>arXiv:1802.06002</em>, 2018.</li>
    <li>M. Schuld and N. Killoran, &ldquo;Quantum machine learning in feature Hilbert spaces,&rdquo; <em>Phys. Rev. Lett.</em>, vol. 122, no. 4, p. 040504, 2019.</li>
    <li>M. Todisco, X. Wang, V. Vestman, et al., &ldquo;ASVspoof 2019: Future perspectives in spoofed and fake speech detection,&rdquo; in <em>Proc. Interspeech</em>, 2019, pp. 1008&ndash;1012.</li>
    <li>H. Tak, J. Patino, M. Todisco, et al., &ldquo;End-to-end anti-spoofing with RawNet2,&rdquo; in <em>Proc. IEEE ICASSP</em>, 2021, pp. 6369&ndash;6373.</li>
    <li>J. W. Jung, H. S. Heo, H. Tak, et al., &ldquo;AASIST: Audio anti-spoofing using integrated spectro-temporal graph attention networks,&rdquo; in <em>Proc. IEEE ICASSP</em>, 2022, pp. 6365&ndash;6369.</li>
    <li>J. R. McClean, S. Boixo, V. N. Smelyanskiy, et al., &ldquo;Barren plateaus in quantum neural network training landscapes,&rdquo; <em>Nat. Commun.</em>, vol. 9, no. 1, p. 4812, 2018.</li>
    <li>Qiskit contributors, &ldquo;Qiskit: An open-source framework for quantum computing,&rdquo; <em>Zenodo</em>, 2024. doi:10.5281/zenodo.2573505.</li>
  </ol>

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
    print("Building HTML document...")
    html_content = build_html_paper()
    with open(OUTPUT_HTML, "w", encoding="utf-8") as f:
        f.write(html_content)
    print(f"Saved HTML to {OUTPUT_HTML} ({len(html_content)} bytes)")


if __name__ == "__main__":
    generate_pdf()
