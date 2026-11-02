"""Classical ML module: Lightweight PyTorch MLP on CPU and NVIDIA CUDA GPU."""

import time
from pathlib import Path
from typing import Dict, Any, Tuple, Optional

import numpy as np

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config

try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import TensorDataset, DataLoader
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


if TORCH_AVAILABLE:
    class VoiceSpoofMLP(nn.Module):
        """Lightweight Multi-Layer Perceptron for voice anti-spoofing."""

        def __init__(
            self,
            input_dim: int = config.TOTAL_FEATURE_DIM,
            hidden_1: int = config.CLASSICAL_HIDDEN_1,
            hidden_2: int = config.CLASSICAL_HIDDEN_2,
            dropout_rate: float = config.DROPOUT_RATE,
            num_classes: int = 2,
        ):
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(input_dim, hidden_1),
                nn.BatchNorm1d(hidden_1),
                nn.ReLU(),
                nn.Dropout(dropout_rate),
                nn.Linear(hidden_1, hidden_2),
                nn.BatchNorm1d(hidden_2),
                nn.ReLU(),
                nn.Dropout(dropout_rate),
                nn.Linear(hidden_2, num_classes),
            )

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return self.net(x)
else:
    class VoiceSpoofMLP:
        pass


def get_available_devices() -> Dict[str, Any]:
    """Check availability of CPU and CUDA GPU devices."""
    if not TORCH_AVAILABLE:
        return {"cpu": False, "cuda": False, "gpu_name": None}
    cuda_avail = torch.cuda.is_available()
    gpu_name = torch.cuda.get_device_name(0) if cuda_avail else None
    return {
        "cpu": True,
        "cuda": cuda_avail,
        "gpu_name": gpu_name,
    }


def train_mlp_model(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    device_name: str = "cpu",
    epochs: int = config.CLASSICAL_EPOCHS,
    batch_size: int = config.CLASSICAL_BATCH_SIZE,
    lr: float = config.CLASSICAL_LR,
    save_path: Optional[Path] = None,
) -> Dict[str, Any]:
    """Train VoiceSpoofMLP on specified device ('cpu' or 'cuda') and record metrics."""
    if not TORCH_AVAILABLE:
        raise RuntimeError("PyTorch is not installed.")

    device = torch.device(device_name)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA device requested but torch.cuda.is_available() is False.")

    input_dim = X_train.shape[1]
    model = VoiceSpoofMLP(input_dim=input_dim).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)

    # Prepare PyTorch datasets
    train_dataset = TensorDataset(
        torch.tensor(X_train, dtype=torch.float32),
        torch.tensor(y_train, dtype=torch.long),
    )
    val_dataset = TensorDataset(
        torch.tensor(X_val, dtype=torch.float32),
        torch.tensor(y_val, dtype=torch.long),
    )

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    history = {"train_loss": [], "val_loss": [], "val_acc": []}

    # Time training with proper CUDA synchronization
    if device.type == "cuda":
        torch.cuda.synchronize()
    start_train_time = time.perf_counter()

    best_val_loss = float("inf")
    best_weights = None

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        for batch_x, batch_y in train_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            optimizer.zero_grad()
            outputs = model(batch_x)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * batch_x.size(0)

        epoch_train_loss = running_loss / len(train_dataset)

        # Validation
        model.eval()
        val_loss = 0.0
        correct = 0
        with torch.no_grad():
            for batch_x, batch_y in val_loader:
                batch_x, batch_y = batch_x.to(device), batch_y.to(device)
                outputs = model(batch_x)
                loss = criterion(outputs, batch_y)
                val_loss += loss.item() * batch_x.size(0)
                preds = torch.argmax(outputs, dim=1)
                correct += (preds == batch_y).sum().item()

        epoch_val_loss = val_loss / len(val_dataset)
        epoch_val_acc = correct / len(val_dataset)

        history["train_loss"].append(epoch_train_loss)
        history["val_loss"].append(epoch_val_loss)
        history["val_acc"].append(epoch_val_acc)

        if epoch_val_loss < best_val_loss:
            best_val_loss = epoch_val_loss
            best_weights = model.state_dict().copy()

    if device.type == "cuda":
        torch.cuda.synchronize()
    total_train_time = time.perf_counter() - start_train_time

    if best_weights is not None:
        model.load_state_dict(best_weights)

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        torch.save(model.state_dict(), save_path)
        print(f"Saved {device_name.upper()} model weights to {save_path}")

    # Benchmark single-sample and batch inference latency
    model.eval()
    benchmark_x = torch.tensor(X_val[:1], dtype=torch.float32).to(device)
    # Warmup
    with torch.no_grad():
        for _ in range(10):
            _ = model(benchmark_x)
        if device.type == "cuda":
            torch.cuda.synchronize()

    # Latency test (averaged over 100 runs)
    runs = 100
    if device.type == "cuda":
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    with torch.no_grad():
        for _ in range(runs):
            _ = model(benchmark_x)
            if device.type == "cuda":
                torch.cuda.synchronize()
    avg_inference_ms = ((time.perf_counter() - t0) / runs) * 1000.0

    return {
        "model": model,
        "device": device_name,
        "device_name": torch.cuda.get_device_name(0) if device.type == "cuda" else "CPU",
        "training_time_sec": float(total_train_time),
        "inference_latency_ms": float(avg_inference_ms),
        "final_train_loss": float(history["train_loss"][-1]),
        "final_val_loss": float(history["val_loss"][-1]),
        "best_val_acc": float(max(history["val_acc"])),
        "history": history,
    }


def predict_single_sample(
    model: Any,
    features: np.ndarray,
    device_name: str = "cpu",
) -> Dict[str, Any]:
    """Inference for a single feature vector."""
    if not TORCH_AVAILABLE:
        raise RuntimeError("PyTorch is not available.")

    device = torch.device(device_name)
    model.eval()
    model.to(device)

    x = torch.tensor(features, dtype=torch.float32)
    if x.ndim == 1:
        x = x.unsqueeze(0)
    x = x.to(device)

    if device.type == "cuda":
        torch.cuda.synchronize()
    t0 = time.perf_counter()

    with torch.no_grad():
        logits = model(x)
        probs = torch.softmax(logits, dim=1).cpu().numpy()[0]
        pred_label = int(np.argmax(probs))

    if device.type == "cuda":
        torch.cuda.synchronize()
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
        "device": device_name,
    }


def predict_batch(
    model: Any,
    X: np.ndarray,
    device_name: str = "cpu",
) -> Tuple[np.ndarray, np.ndarray, float]:
    """Batch inference returning predictions, probabilities, and total elapsed seconds."""
    device = torch.device(device_name)
    model.eval()
    model.to(device)

    x = torch.tensor(X, dtype=torch.float32).to(device)

    if device.type == "cuda":
        torch.cuda.synchronize()
    t0 = time.perf_counter()

    with torch.no_grad():
        logits = model(x)
        probs = torch.softmax(logits, dim=1).cpu().numpy()
        preds = np.argmax(probs, axis=1)

    if device.type == "cuda":
        torch.cuda.synchronize()
    total_time = time.perf_counter() - t0

    return preds, probs, total_time
