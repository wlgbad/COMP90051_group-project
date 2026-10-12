"""FT-Transformer numerical feature tokenizer and training routines.

Architecture, optimiser, seed and early-stopping semantics match the
completed FT experiments. No cross-validation or dataset handling here.
"""
from __future__ import annotations
import math
import random
import time
from typing import Any
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from shared.shared_metrics import classification_metrics_from_scratch
from shared.shared_protocol import N_CLASSES

def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed % (2**32 - 1))
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def device_from_choice(choice: str) -> torch.device:
    if choice == "auto":
        if torch.cuda.is_available():
            choice = "cuda"
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            choice = "mps"
        else:
            choice = "cpu"
    if choice == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested, but PyTorch cannot see an NVIDIA GPU.")
    if choice == "mps" and not (
        hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
    ):
        raise RuntimeError("MPS requested, but is unavailable.")
    return torch.device(choice)


class FTTransformer(nn.Module):
    """Numerical feature tokenizer + CLS-token Transformer encoder.

    Each of the 396 numerical features contributes a token. This is a full
    feature-token transformer, not an MLP relabeled as a transformer.
    """
    def __init__(self, n_features: int, n_classes: int, d_token: int,
                 n_heads: int, n_layers: int, ffn_mult: int, dropout: float):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(n_features, d_token))
        self.bias = nn.Parameter(torch.zeros(n_features, d_token))
        self.cls = nn.Parameter(torch.zeros(1, 1, d_token))
        nn.init.normal_(self.weight, std=0.02)
        nn.init.normal_(self.cls, std=0.02)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_token,
            nhead=n_heads,
            dim_feedforward=d_token * ffn_mult,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(
            encoder_layer, num_layers=n_layers, enable_nested_tensor=False
        )
        self.norm = nn.LayerNorm(d_token)
        self.head = nn.Linear(d_token, n_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, 396), tokens: (batch, 397, d_token)
        tokens = x.unsqueeze(-1) * self.weight.unsqueeze(0) + self.bias.unsqueeze(0)
        cls = self.cls.expand(x.shape[0], -1, -1)
        encoded = self.encoder(torch.cat((cls, tokens), dim=1))
        return self.head(self.norm(encoded[:, 0]))


def new_model(cfg: argparse.Namespace, n_features: int) -> FTTransformer:
    return FTTransformer(
        n_features=n_features,
        n_classes=N_CLASSES,
        d_token=cfg.d_token,
        n_heads=cfg.n_heads,
        n_layers=cfg.n_layers,
        ffn_mult=cfg.ffn_mult,
        dropout=cfg.dropout,
    )


def loader_for(X: np.ndarray, y: np.ndarray, batch_size: int, shuffle: bool,
               seed: int) -> DataLoader:
    ds = TensorDataset(torch.from_numpy(np.ascontiguousarray(X, dtype=np.float32)),
                       torch.from_numpy(np.ascontiguousarray(y, dtype=np.int64)))
    g = torch.Generator().manual_seed(seed)
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle, num_workers=0,
                      generator=g, pin_memory=torch.cuda.is_available())


def predict(model: nn.Module, X: np.ndarray, batch_size: int,
            device: torch.device) -> np.ndarray:
    model.eval()
    labels: list[np.ndarray] = []
    with torch.inference_mode():
        for start in range(0, len(X), batch_size):
            xb = torch.from_numpy(np.ascontiguousarray(X[start:start + batch_size],
                                                        dtype=np.float32)).to(device)
            labels.append(model(xb).argmax(dim=1).cpu().numpy())
    return np.concatenate(labels).astype(np.int64) if labels else np.empty(0, dtype=np.int64)


def one_epoch(model: nn.Module, data_loader: DataLoader,
              optimizer: torch.optim.Optimizer, device: torch.device) -> float:
    model.train()
    total_loss = 0.0
    seen = 0
    for xb, yb in data_loader:
        xb = xb.to(device, non_blocking=True)
        yb = yb.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        loss = nn.functional.cross_entropy(model(xb), yb)
        loss.backward()
        optimizer.step()
        total_loss += float(loss.item()) * len(yb)
        seen += len(yb)
    return total_loss / max(seen, 1)


def inner_fit(X_train: np.ndarray, y_train: np.ndarray,
              X_val: np.ndarray, y_val: np.ndarray,
              cfg: argparse.Namespace, lr: float, seed: int,
              device: torch.device) -> dict[str, Any]:
    set_seed(seed)
    model = new_model(cfg, X_train.shape[1]).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr,
                                  weight_decay=cfg.weight_decay)
    data_loader = loader_for(X_train, y_train, cfg.batch_size, True, seed)
    best_score = -math.inf
    best_epoch = 1
    best_accuracy = 0.0
    no_improvement = 0
    t0 = time.perf_counter()
    for epoch in range(1, cfg.epochs + 1):
        loss = one_epoch(model, data_loader, optimizer, device)
        pred = predict(model, X_val, cfg.batch_size, device)
        metrics, _ = classification_metrics_from_scratch(y_val, pred, N_CLASSES)
        score = metrics["macro_f1"]
        if score > best_score + 1e-10:
            best_score = score
            best_accuracy = metrics["accuracy"]
            best_epoch = epoch
            no_improvement = 0
        else:
            no_improvement += 1
        if epoch == 1 or epoch % 10 == 0 or no_improvement >= cfg.patience:
            print(f"    epoch {epoch:3d}: loss={loss:.4f}, val_f1={score:.4f}, "
                  f"best_f1={best_score:.4f} @ {best_epoch}", flush=True)
        if no_improvement >= cfg.patience:
            break
    return {
        "best_macro_f1": float(best_score), "accuracy_at_best_f1": float(best_accuracy),
        "best_epoch": int(best_epoch), "epochs_executed": int(epoch),
        "runtime_seconds": round(time.perf_counter() - t0, 2),
    }


def final_fit(X_train: np.ndarray, y_train: np.ndarray,
              cfg: argparse.Namespace, lr: float, epochs: int,
              seed: int, device: torch.device) -> nn.Module:
    set_seed(seed)
    model = new_model(cfg, X_train.shape[1]).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr,
                                  weight_decay=cfg.weight_decay)
    data_loader = loader_for(X_train, y_train, cfg.batch_size, True, seed)
    for epoch in range(1, epochs + 1):
        loss = one_epoch(model, data_loader, optimizer, device)
        if epoch == 1 or epoch % 10 == 0 or epoch == epochs:
            print(f"    full-train epoch {epoch:3d}/{epochs}: loss={loss:.4f}", flush=True)
    return model


