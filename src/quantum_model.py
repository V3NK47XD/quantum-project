"""Genuine Variational Quantum Classifier (VQC) implementation using Qiskit Aer."""

import time
from pathlib import Path
from typing import Dict, Any, Tuple, Optional, Callable, List

import numpy as np
import matplotlib.pyplot as plt

import qiskit
from qiskit import QuantumCircuit
from qiskit.circuit.library import zz_feature_map, real_amplitudes, efficient_su2
from qiskit_aer.primitives import SamplerV2
from qiskit_algorithms.optimizers import COBYLA, SPSA
from qiskit_machine_learning.algorithms import VQC

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config


def build_custom_rotation_feature_map(num_qubits: int = 4) -> QuantumCircuit:
    """Build a custom quantum feature map with explicit single-qubit Ry, Rz rotations and CNOTs."""
    qc = QuantumCircuit(num_qubits, name="CustomRotationMap")
    x_params = [qiskit.circuit.Parameter(f"x[{i}]") for i in range(num_qubits)]

    # Initial superposition
    for q in range(num_qubits):
        qc.h(q)

    # First rotation layer: Ry(x[i])
    for q in range(num_qubits):
        qc.ry(x_params[q], q)

    # Entangling CNOT ladder
    for q in range(num_qubits - 1):
        qc.cx(q, q + 1)
    qc.cx(num_qubits - 1, 0)

    # Second rotation layer: Rz(x[i])
    for q in range(num_qubits):
        qc.rz(x_params[q], q)

    return qc


def build_quantum_circuit(
    num_qubits: int = config.N_QUBITS,
    feature_map_type: str = "ZZFeatureMap",
    ansatz_type: str = "RealAmplitudes",
    ansatz_reps: int = config.ANSATZ_REPS,
) -> Tuple[QuantumCircuit, QuantumCircuit, QuantumCircuit]:
    """
    Construct the quantum feature map, trainable ansatz with CNOTs, and combined circuit.
    """
    # 1. Feature Map
    if feature_map_type == "CustomRotation":
        fm = build_custom_rotation_feature_map(num_qubits=num_qubits)
    else:  # default ZZFeatureMap
        fm = zz_feature_map(
            feature_dimension=num_qubits,
            reps=1,
            entanglement="linear",
        )

    # 2. Trainable Ansatz with CNOT entanglement
    if ansatz_type == "EfficientSU2":
        ansatz = efficient_su2(
            num_qubits=num_qubits,
            reps=ansatz_reps,
            entanglement="linear",
            su2_gates=["ry", "rz"],
        )
    else:  # default RealAmplitudes
        ansatz = real_amplitudes(
            num_qubits=num_qubits,
            reps=ansatz_reps,
            entanglement="linear",
        )

    # 3. Composed Circuit
    composed_qc = fm.compose(ansatz)
    return fm, ansatz, composed_qc


class QuantumVoiceClassifier:
    """Variational Quantum Classifier for audio spoof detection on Qiskit Aer."""

    def __init__(
        self,
        num_qubits: int = config.N_QUBITS,
        feature_map_type: str = config.FEATURE_MAP_TYPE,
        ansatz_type: str = "RealAmplitudes",
        ansatz_reps: int = config.ANSATZ_REPS,
        maxiter: int = config.QUANTUM_MAX_ITER,
        shots: int = config.QUANTUM_SHOTS,
        seed: int = config.QUANTUM_SEED,
    ):
        self.num_qubits = num_qubits
        self.feature_map_type = feature_map_type
        self.ansatz_type = ansatz_type
        self.ansatz_reps = ansatz_reps
        self.maxiter = maxiter
        self.shots = shots
        self.seed = seed

        # Build circuits
        self.feature_map, self.ansatz, self.composed_circuit = build_quantum_circuit(
            num_qubits=self.num_qubits,
            feature_map_type=self.feature_map_type,
            ansatz_type=self.ansatz_type,
            ansatz_reps=self.ansatz_reps,
        )

        # Aer Sampler Primitive
        self.sampler = SamplerV2(options={"run_options": {"shots": self.shots, "seed": self.seed}})

        # Optimizer with callback tracking
        self.cost_history: List[float] = []

        def cobyla_callback(weights):
            pass

        self.optimizer = COBYLA(maxiter=self.maxiter, callback=cobyla_callback)

        # Qiskit ML VQC
        self.vqc = VQC(
            feature_map=self.feature_map,
            ansatz=self.ansatz,
            optimizer=self.optimizer,
            sampler=self.sampler,
            callback=self._store_cost,
        )

        self.is_fitted = False
        self.training_time_sec = 0.0
        self.optimal_weights = None

    def _store_cost(self, weights, cost):
        self.cost_history.append(float(cost))

    def fit(self, X_train: np.ndarray, y_train: np.ndarray) -> "QuantumVoiceClassifier":
        """Train the VQC using classical optimization over Aer quantum simulation."""
        self.cost_history.clear()
        t0 = time.perf_counter()
        print(f"Training Quantum VQC on Qiskit Aer (Samples: {len(X_train)}, MaxIter: {self.maxiter})...")

        # Fit VQC
        self.vqc.fit(X_train, y_train)

        self.training_time_sec = float(time.perf_counter() - t0)
        self.is_fitted = True
        self.optimal_weights = self.vqc.weights
        print(f"Quantum VQC training complete in {self.training_time_sec:.2f}s! Final cost: {self.cost_history[-1] if self.cost_history else 'N/A'}")
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict binary class labels (0: Bonafide, 1: Spoof)."""
        if not self.is_fitted:
            raise RuntimeError("VQC model is not trained yet.")
        raw_preds = self.vqc.predict(X)
        if raw_preds.ndim > 1:
            raw_preds = np.argmax(raw_preds, axis=1)
        return raw_preds.astype(int)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Compute class probabilities via quantum circuit measurement counts."""
        if not self.is_fitted:
            raise RuntimeError("VQC model is not trained yet.")
        # VQC neural network forward pass gives measurement probabilities
        raw_probs = self.vqc.neural_network.forward(X, self.vqc.weights)
        # In VQC, raw_probs has shape (N, 2) when 2 classes are mapped
        # Normalize just in case
        probs = np.clip(raw_probs, 1e-6, 1.0)
        probs = probs / np.sum(probs, axis=1, keepdims=True)
        return probs

    def predict_single(self, x_4d: np.ndarray) -> Dict[str, Any]:
        """Perform quantum inference on a single 4-dimensional sample and measure latency."""
        if not self.is_fitted:
            raise RuntimeError("VQC model is not trained yet.")

        x = np.asarray(x_4d).reshape(1, -1)
        t0 = time.perf_counter()
        probs = self.predict_proba(x)[0]
        pred_label = int(np.argmax(probs))
        latency_ms = (time.perf_counter() - t0) * 1000.0

        return {
            "predicted_label": pred_label,
            "class_name": config.LABEL_NAMES[pred_label],
            "confidence": float(probs[pred_label]),
            "probabilities": {
                config.LABEL_NAMES[0]: float(probs[0]),
                config.LABEL_NAMES[1]: float(probs[1]),
            },
            "latency_ms": float(latency_ms),
            "backend": "Qiskit AerSimulator (Statevector/Shot-based)",
            "num_qubits": self.num_qubits,
        }

    def draw_circuit(self, output: str = "mpl") -> Any:
        """Render the complete quantum circuit diagram."""
        # Unroll circuit for visual inspection
        composed = self.feature_map.compose(self.ansatz)
        composed.measure_all()
        if output == "text":
            return composed.draw(output="text")
        return composed.draw(output="mpl")

    def save_weights(self, path: Path) -> None:
        """Save trained variational parameters."""
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        np.save(path, self.vqc.weights)
        print(f"Saved Quantum VQC weights to {path}")

    def load_weights(self, path: Path) -> None:
        """Load trained variational parameters."""
        weights = np.load(path)
        self.optimal_weights = weights
        self.vqc.warm_start = True
        self.vqc._initial_point = weights
        self.vqc._fit_result = type("DummyFitResult", (), {"x": weights})()
        self.is_fitted = True
        print(f"Loaded Quantum VQC weights from {path}")
