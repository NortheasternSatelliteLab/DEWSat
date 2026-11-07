"""Temporal networks (DESIGN.md section 5): bilstm, resbilstm, gru, lstm, tcn; training and inference.

The Bi-LSTM layout is the lab's DroughtBiLSTM (as in the capability run's neural.py, commit e13e7a4): final states of
both directions plus the masked mean of the outputs -> LayerNorm -> Linear -> GELU -> Dropout -> Linear, logits out.
Inputs are (n, steps, channels); channel OBS_CH is the observed flag (1/0), used for the masked mean and for masking
augmentation. The residual model adds a base logit computed outside the network (the job's LR), so with its last
layer at zero it starts exactly at the LR.
"""

from __future__ import annotations

import math
import time

import numpy as np
import torch
from torch import nn


class Encoder(nn.Module):
    def __init__(self, family: str, n_in: int, hidden: int, layers: int, dropout: float, bidir: bool):
        super().__init__()
        self.family = family
        if family in ("bilstm", "resbilstm", "lstm"):
            self.rnn = nn.LSTM(n_in, hidden, num_layers=layers, batch_first=True, bidirectional=bidir,
                               dropout=dropout if layers > 1 else 0.0)
        elif family == "gru":
            self.rnn = nn.GRU(n_in, hidden, num_layers=layers, batch_first=True, bidirectional=bidir,
                              dropout=dropout if layers > 1 else 0.0)
        else:
            raise ValueError(family)
        self.bidir = bidir
        self.out_dim = hidden * (2 if bidir else 1)

    def forward(self, x, mask):
        out, h = self.rnn(x)
        h = h[0] if isinstance(h, tuple) else h            # LSTM: (h, c)
        final = torch.cat((h[-2], h[-1]), dim=1) if self.bidir else h[-1]
        mean = (out * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1.0)
        return torch.cat((final, mean), dim=1)


class CausalBlock(nn.Module):
    def __init__(self, ch: int, dilations=(1, 2, 4), kernel: int = 3, dropout: float = 0.0):
        super().__init__()
        self.convs = nn.ModuleList(nn.Conv1d(ch, ch, kernel, dilation=d) for d in dilations)
        self.pads = [(kernel - 1) * d for d in dilations]
        self.norm = nn.ModuleList(nn.GroupNorm(1, ch) for _ in dilations)
        self.drop = nn.Dropout(dropout)
        self.act = nn.GELU()

    def forward(self, x):                                   # x: (n, ch, steps)
        y = x
        for conv, pad, norm in zip(self.convs, self.pads, self.norm):
            y = self.drop(self.act(norm(conv(nn.functional.pad(y, (pad, 0))))))
        return x + y


class TCNEncoder(nn.Module):
    def __init__(self, n_in: int, hidden: int, layers: int, dropout: float):
        super().__init__()
        self.inp = nn.Conv1d(n_in, hidden, 1)
        self.blocks = nn.Sequential(*[CausalBlock(hidden, dropout=dropout) for _ in range(layers)])
        self.out_dim = hidden

    def forward(self, x, mask):
        y = self.blocks(self.inp(x.transpose(1, 2))).transpose(1, 2)    # (n, steps, hidden)
        last = y[:, -1]
        mean = (y * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1.0)
        return torch.cat((last, mean), dim=1)


class SeqNet(nn.Module):
    def __init__(self, family: str, n_in: int, n_heads: int, hidden: int, layers: int = 1, dropout: float = 0.2,
                 bidir: bool | None = None, obs_ch: int = 7):
        super().__init__()
        self.family = family
        self.obs_ch = obs_ch
        if family == "tcn":
            self.enc = TCNEncoder(n_in, hidden, layers, dropout)
        else:
            if bidir is None:
                bidir = family in ("bilstm", "resbilstm")
            self.enc = Encoder(family, n_in, hidden, layers, dropout, bidir)
        d = 2 * self.enc.out_dim
        mlp = 2 * hidden
        self.head = nn.Sequential(nn.LayerNorm(d), nn.Linear(d, mlp), nn.GELU(), nn.Dropout(dropout),
                                  nn.Linear(mlp, n_heads))
        if family == "resbilstm":
            nn.init.zeros_(self.head[-1].weight)
            nn.init.zeros_(self.head[-1].bias)

    def forward(self, x):
        mask = x[:, :, self.obs_ch:self.obs_ch + 1]
        return self.head(self.enc(x, mask))


def count_parameters(model) -> int:
    return int(sum(p.numel() for p in model.parameters()))


def setup_torch(seed: int, device: str, threads: int):
    torch.manual_seed(seed)
    np.random.seed(seed % (2 ** 32))
    torch.set_num_threads(max(1, threads))
    torch.use_deterministic_algorithms(device == "cpu", warn_only=True)


def _lr_at(epoch: int, epochs: int, lr: float, floor: float = 0.05) -> float:
    progress = (epoch - 1) / max(1, epochs - 1)
    lo = lr * floor
    return lo + 0.5 * (lr - lo) * (1 + math.cos(math.pi * progress))


@torch.inference_mode()
def predict_logits(model, X: np.ndarray, base=None, device="cpu", batch: int = 8192) -> np.ndarray:
    """Logits (n, heads); base (n, heads) logits are added for the residual model."""
    model.eval()
    out = []
    for s in range(0, len(X), batch):
        xb = torch.from_numpy(X[s:s + batch]).to(device)
        out.append(model(xb).float().cpu().numpy())
    z = np.concatenate(out) if out else np.zeros((0, model.head[-1].out_features), np.float32)
    if base is not None:
        z = z + base
    return z


AGE_CH, OBS_CH = 8, 7
ZERO_ON_HIDE = np.array([1, 1, 1, 1, 1, 0, 0, 1, 0], np.float32)   # channels cleared when a month is hidden


def recompute_age(xb: torch.Tensor) -> torch.Tensor:
    """Rebuild the months-since-last-observation channel (scaled by 1/12) from the observed flags."""
    obs = xb[:, :, OBS_CH] > 0.5
    steps = xb.shape[1]
    age = torch.zeros(obs.shape, dtype=xb.dtype, device=xb.device)
    cur = torch.full((obs.shape[0],), -1.0, dtype=xb.dtype, device=xb.device)    # -1: nothing seen yet
    for t in range(steps):
        seen = cur >= 0
        a = torch.where(obs[:, t], torch.zeros_like(cur),
                        torch.where(seen, cur + 1, torch.full_like(cur, float(min(12, t + 1)))))
        age[:, t] = a
        cur = torch.where(obs[:, t], torch.zeros_like(cur), torch.where(seen, cur + 1, cur))
    out = xb.clone()
    out[:, :, AGE_CH] = age / 12.0
    return out


def latest_age_logit(coef: dict, xb: torch.Tensor, vb: torch.Tensor) -> torch.Tensor:
    """LR on the latest observed VCI + its age (raw units), per head; base-rate logit if nothing is observed."""
    obs = xb[:, :, OBS_CH] > 0.5
    steps = xb.shape[1]
    idx = torch.arange(steps, device=xb.device).expand_as(obs)
    pos = torch.where(obs, idx, torch.full_like(idx, -1)).max(dim=1).values
    has = pos >= 0
    posc = pos.clamp(min=0)
    v = vb.gather(1, posc[:, None])[:, 0]
    age = (steps - 1 - posc).to(vb.dtype)
    z = coef["a"][None, :] + coef["b"][None, :] * v[:, None] + coef["c"][None, :] * age[:, None]
    return torch.where(has[:, None], z, coef["a0"][None, :].expand_as(z))


def train(cfg: dict, data: dict, seed: int, device: str = "cpu", threads: int = 3, log=None) -> dict:
    """One run with a fixed number of epochs and a cosine schedule (no early stopping).

    data: X_fit (n, steps, 9) float32; Y_fit (n, heads) {0,1}; M_fit (n, heads) bool; hide_ok (steps,) bool = months
          the masking augmentation may hide; optional base_fit (n, heads) fixed base logits (residual model, jobs
          1/3: LR on VCI(T), unaffected by masking); optional V_fit (n, steps) raw VCI and base_coef (dict of per-head
          tensors a, b, c, a0) for job 2's residual base, recomputed after masking; X_eval: list of
          (name, X, base or None) predicted at the end.
    cfg: family, hidden, layers, lr, weight_decay, dropout, mask_p (0 = off), epochs, bidir (gru), batch.
    """
    setup_torch(seed, device, threads)
    rng = np.random.default_rng(seed)
    n_heads = data["Y_fit"].shape[1]
    n_in = data["X_fit"].shape[2]
    model = SeqNet(cfg["family"], n_in, n_heads, cfg["hidden"], cfg.get("layers", 1), cfg.get("dropout", 0.2),
                   cfg.get("bidir"), data.get("obs_ch", OBS_CH)).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"])
    lossf = nn.BCEWithLogitsLoss(reduction="none")
    X = torch.from_numpy(data["X_fit"]).to(device)
    Y = torch.from_numpy(data["Y_fit"].astype(np.float32)).to(device)
    M = torch.from_numpy(data["M_fit"].astype(np.float32)).to(device)
    zm = data.get("zero_on_hide")
    zero_mask = torch.from_numpy(np.asarray(ZERO_ON_HIDE if zm is None else zm, np.float32)).to(device)
    hide_ok = torch.from_numpy(np.asarray(data["hide_ok"], bool)).to(device)
    B_fit = None if data.get("base_fit") is None else torch.from_numpy(data["base_fit"].astype(np.float32)).to(device)
    V = None if data.get("V_fit") is None else torch.from_numpy(data["V_fit"].astype(np.float32)).to(device)
    coef = None
    if data.get("base_coef") is not None:
        coef = {k: torch.tensor(np.asarray(v, np.float32), device=device) for k, v in data["base_coef"].items()}
    n = len(X)
    steps = X.shape[1]
    batch = int(cfg.get("batch", 512))
    epochs = int(cfg["epochs"])
    p = float(cfg.get("mask_p", 0.0))
    history = []
    t_start = time.perf_counter()
    gen = torch.Generator(device="cpu").manual_seed(seed)
    for epoch in range(1, epochs + 1):
        t0 = time.perf_counter()
        rate = _lr_at(epoch, epochs, cfg["lr"])
        for g in opt.param_groups:
            g["lr"] = rate
        model.train()
        order = torch.from_numpy(rng.permutation(n)).to(device)
        total, seen = 0.0, 0.0
        for s in range(0, n, batch):
            bi = order[s:s + batch]
            xb = X.index_select(0, bi)
            if p > 0:
                u = torch.rand((len(bi), steps), generator=gen).to(device)
                hide = (u < p) & hide_ok[None, :] & (xb[:, :, OBS_CH] > 0.5)
                if bool(hide.any()):
                    xb = xb * (1.0 - hide[:, :, None].float() * zero_mask)
                    xb = recompute_age(xb)
            yb, mb = Y.index_select(0, bi), M.index_select(0, bi)
            opt.zero_grad(set_to_none=True)
            z = model(xb)
            if coef is not None:
                z = z + latest_age_logit(coef, xb, V.index_select(0, bi))
            elif B_fit is not None:
                z = z + B_fit.index_select(0, bi)
            loss = (lossf(z, yb) * mb).sum() / mb.sum().clamp(min=1.0)
            if not torch.isfinite(loss):
                raise RuntimeError("non-finite training loss")
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            w = float(mb.sum())
            total += float(loss.detach()) * w
            seen += w
        rec = {"epoch": epoch, "lr": rate, "train_loss": total / max(seen, 1),
               "seconds": round(time.perf_counter() - t0, 2)}
        history.append(rec)
        if log:
            log(f"    {cfg['family']} seed {seed} epoch {epoch:02d}/{epochs} train {rec['train_loss']:.4f} "
                f"{rec['seconds']:.1f}s")
    preds = {}
    for name, Xe, be in data.get("X_eval", []):
        preds[name] = predict_logits(model, Xe, be, device)
    return {"model": model, "history": history, "seconds": round(time.perf_counter() - t_start, 1),
            "n_params": count_parameters(model), "logits": preds,
            "state": {k: v.detach().cpu() for k, v in model.state_dict().items()}}
