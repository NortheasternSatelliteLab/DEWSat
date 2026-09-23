"""The Bi-LSTM, its losses, and inference helpers.

The core of v1's model was right and is kept: one bidirectional LSTM whose two
final top-layer hidden states summarize the window, then a small bounded head.
What is added is an eighth input channel marking observed months, an optional
mask-aware pooling path, and an export-friendly forward pass with no data
dependent control flow.
"""
import numpy as np
import torch
from torch import nn

from .contract import MODEL_FEATURES

POOLINGS = ("final", "final_masked_mean")


class DroughtBiLSTM(nn.Module):
    """One bounded prediction per window of monthly features."""

    def __init__(self, hidden_size=32, num_layers=1, dropout=0.0, mlp_hidden=64,
                 pool="final", sequence_length=12):
        super().__init__()
        if pool not in POOLINGS:
            raise ValueError(f"pool must be one of {list(POOLINGS)}.")
        self.config = {"hidden_size": hidden_size, "num_layers": num_layers, "dropout": dropout,
                       "mlp_hidden": mlp_hidden, "pool": pool, "sequence_length": sequence_length}
        self.pool = pool
        self.sequence_length = sequence_length
        self.lstm = nn.LSTM(input_size=len(MODEL_FEATURES), hidden_size=hidden_size,
                            num_layers=num_layers, batch_first=True, bidirectional=True,
                            dropout=dropout if num_layers > 1 else 0.0)
        summary_size = 2 * hidden_size * (2 if pool == "final_masked_mean" else 1)
        self.head = nn.Sequential(
            nn.LayerNorm(summary_size),
            nn.Linear(summary_size, mlp_hidden), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(mlp_hidden, 1), nn.Sigmoid())

    def forward(self, x):
        if x.dim() != 3 or x.shape[1] != self.sequence_length or x.shape[2] != len(MODEL_FEATURES):
            raise ValueError(f"Model expects [batch,{self.sequence_length},{len(MODEL_FEATURES)}], "
                             f"got {list(x.shape)}.")
        outputs, (hidden, _) = self.lstm(x)
        # The top layer's final forward and backward states each summarize the
        # whole window from one direction; the last timestep alone would not.
        summary = torch.cat((hidden[-2], hidden[-1]), dim=1)
        if self.pool == "final_masked_mean":
            mask = x[:, :, -1:]                      # the observed channel
            total = mask.sum(dim=1).clamp(min=1.0)   # windows always keep >= 2 real months
            summary = torch.cat((summary, (outputs * mask).sum(dim=1) / total), dim=1)
        return self.head(summary).squeeze(-1)  # squeeze(-1) keeps the batch axis at batch size 1


def weighted_huber(prediction, target, delta=0.1, dry_weight=2.0):
    """Huber loss with dry months upweighted by 1 + dry_weight*(1 - target)."""
    if prediction.shape != target.shape:
        raise ValueError("Prediction and target shapes must match; silent broadcasting hides bugs.")
    losses = nn.functional.huber_loss(prediction, target, delta=delta, reduction="none")
    weights = 1 + dry_weight * (1 - target)
    return (weights * losses).sum() / weights.sum()


def distillation_loss(student, teacher, target, alpha=0.5, delta=0.1, dry_weight=2.0):
    """Blend the supervised loss with agreement to a frozen teacher's outputs.

    Regression distillation needs no temperature: the teacher's continuous
    output already carries more information per example than the label alone.
    """
    if not 0 <= alpha <= 1:
        raise ValueError("alpha must be in [0,1]; 1 ignores the teacher entirely.")
    hard = weighted_huber(student, target, delta, dry_weight)
    soft = weighted_huber(student, teacher.detach(), delta, dry_weight)
    return alpha * hard + (1 - alpha) * soft, hard.detach(), soft.detach()


def predict_batches(model, table, windows, index, scaler, device="cpu", batch_size=512):
    """Run inference over window indices, building each batch on demand."""
    from .data import build_inputs
    model.eval()
    outputs = []
    with torch.inference_mode():
        for start in range(0, len(index), batch_size):
            chunk = index[start:start + batch_size]
            x = torch.from_numpy(build_inputs(table, windows, chunk, scaler)).to(device)
            outputs.append(model(x).float().cpu().numpy())
    return np.concatenate(outputs) if outputs else np.zeros(0, dtype=np.float32)


def count_parameters(model):
    return int(sum(p.numel() for p in model.parameters()))
