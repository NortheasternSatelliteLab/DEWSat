"""CNN models of DESIGN.md amendment 4: a tile encoder shared by cnn_lr (month T only) and cnn_bilstm (12 months).

Tiles come from cache/tiles (tiles.py): (cells, months, 23, 21) uint8, VCI 0-100, EMPTY (255) = unusable bin; an
unobserved cell-month is an all-EMPTY tile. A window is (cell, T): cnn_lr reads month T's tile, cnn_bilstm months
T-11..T (months before 2016-01 are empty, like the padding of the summary inputs). Encoder input channels: VCI/100
(0 where unusable) and the usable flag. The observed flag and months-since-last-observation per month are the
summary pipeline's (channels 7 and 8 of windows.gather), so every job-1 model sees the same observation pattern.
"""

from __future__ import annotations

import time

import numpy as np
import torch
from torch import nn

import nets

EMPTY = 255


class TileEncoder(nn.Module):
    """conv 3x3 (2 -> w), GELU; conv 3x3 stride 2 (w -> 2w), GELU; [conv 3x3 stride 2 (2w -> 2w), GELU]; then each
    channel's mean over usable output positions and the tile's usable share: 2w + 1 numbers per tile."""

    def __init__(self, width: int, depth: int):
        super().__init__()
        w = int(width)
        layers = [nn.Conv2d(2, w, 3, padding=1), nn.GELU(), nn.Conv2d(w, 2 * w, 3, stride=2, padding=1), nn.GELU()]
        if depth >= 3:
            layers += [nn.Conv2d(2 * w, 2 * w, 3, stride=2, padding=1), nn.GELU()]
        self.net = nn.Sequential(*layers)
        self.n_stride = 2 if depth >= 3 else 1
        self.out_dim = 2 * w + 1

    def forward(self, x):                                   # x: (n, 2, H, W)
        y = self.net(x)
        m = x[:, 1:2]
        for _ in range(self.n_stride):                      # an output position is usable if its window holds one
            m = nn.functional.max_pool2d(m, 3, stride=2, padding=1)
        num = (y * m).sum(dim=(2, 3))
        den = m.sum(dim=(2, 3)).clamp(min=1.0)
        share = x[:, 1].mean(dim=(1, 2))[:, None]
        return torch.cat((num / den, share), dim=1)


class CNNLR(nn.Module):
    """cnn_lr: the encoder on month T's tile, dropout, one linear layer (a logistic regression on learned features)."""

    def __init__(self, width: int, depth: int, dropout: float, n_heads: int = 1):
        super().__init__()
        self.enc = TileEncoder(width, depth)
        self.drop = nn.Dropout(dropout)
        self.out = nn.Linear(self.enc.out_dim, n_heads)

    def forward(self, tiles, obs=None, age=None):           # tiles: (n, 2, H, W)
        return self.out(self.drop(self.enc(tiles)))


class CNNBiLSTM(nn.Module):
    """cnn_bilstm: the encoder on each monthly tile (shared weights) + observed + months-since -> the bilstm encoder and
    head of nets.SeqNet (final states of both directions + masked mean -> LayerNorm -> Linear -> GELU -> Dropout ->
    Linear)."""

    def __init__(self, width: int, depth: int, hidden: int, layers: int, dropout: float, n_heads: int = 1):
        super().__init__()
        self.enc = TileEncoder(width, depth)
        self.rnn = nets.Encoder("bilstm", self.enc.out_dim + 2, hidden, layers, dropout, True)
        d = 2 * self.rnn.out_dim
        self.head = nn.Sequential(nn.LayerNorm(d), nn.Linear(d, 2 * hidden), nn.GELU(), nn.Dropout(dropout),
                                  nn.Linear(2 * hidden, n_heads))

    def forward(self, tiles, obs, age):                     # tiles: (n, steps, 2, H, W); obs, age: (n, steps)
        n, s = tiles.shape[:2]
        f = self.enc(tiles.reshape(n * s, *tiles.shape[2:])).reshape(n, s, -1)
        x = torch.cat((f, obs[..., None], age[..., None]), dim=2)
        return self.head(self.rnn(x, obs[..., None]))


def build(cfg: dict, n_heads: int = 1) -> nn.Module:
    if cfg["family"] == "cnn_lr":
        return CNNLR(cfg["width"], cfg["depth"], cfg["dropout"], n_heads)
    if cfg["family"] == "cnn_bilstm":
        return CNNBiLSTM(cfg["width"], cfg["depth"], cfg["hidden"], cfg.get("layers", 1), cfg["dropout"], n_heads)
    raise ValueError(cfg["family"])


def multiply_adds(cfg: dict, steps: int = 12, h: int = 23, w: int = 21) -> int:
    """Multiply-adds for one prediction (one window), counted layer by layer (convolutions, LSTM, linear layers)."""
    wd, out = int(cfg["width"]), lambda n: (n + 2 - 3) // 2 + 1
    h1, w1 = out(h), out(w)
    macs = h * w * wd * 2 * 9 + h1 * w1 * 2 * wd * wd * 9
    if cfg["depth"] >= 3:
        macs += out(h1) * out(w1) * 2 * wd * 2 * wd * 9
    if cfg["family"] == "cnn_lr":
        return int(macs + (2 * wd + 1))
    hid, n_in = int(cfg["hidden"]), 2 * wd + 3
    lstm = steps * 2 * 4 * hid * (n_in + hid)               # both directions, four gates
    head = 4 * hid * 2 * hid + 2 * hid
    return int(steps * macs + lstm + head)


def window_tiles(tiles: np.ndarray, cells: np.ndarray, T: np.ndarray, steps: int) -> np.ndarray:
    """(n, steps, H, W) uint8 tiles of months T-steps+1..T for cells (EMPTY before the first month)."""
    idx = T[:, None] + np.arange(-steps + 1, 1)[None, :]
    out = tiles[cells[:, None], np.clip(idx, 0, None)]
    out[idx < 0] = EMPTY
    return out


def to_input(t: torch.Tensor) -> torch.Tensor:
    """uint8 tiles (..., H, W) on the device -> float (..., 2, H, W): VCI/100 (0 where unusable), usable flag."""
    usable = t != EMPTY
    v = torch.where(usable, t.float() / 100.0, torch.zeros((), device=t.device))
    return torch.stack((v, usable.float()), dim=-3)


def age_from_obs(obs: torch.Tensor) -> torch.Tensor:
    """Months since the last observation / 12, the rule of windows.fill_age (0 when observed; counts up; from the
    window start, capped at 12, before the first observation)."""
    steps = obs.shape[1]
    age = torch.zeros(obs.shape, dtype=torch.float32, device=obs.device)
    cur = torch.full((obs.shape[0],), -1.0, device=obs.device)
    for t in range(steps):
        seen = cur >= 0
        age[:, t] = torch.where(obs[:, t], torch.zeros_like(cur),
                                torch.where(seen, cur + 1, torch.full_like(cur, float(min(12, t + 1)))))
        cur = torch.where(obs[:, t], torch.zeros_like(cur), torch.where(seen, cur + 1, cur))
    return age / 12.0


class Batches:
    """Window inputs for one split: tiles from the fold's tile array, observed/age from the summary windows."""

    def __init__(self, tiles: np.ndarray, cells: np.ndarray, T: np.ndarray, X: np.ndarray, steps: int):
        self.tiles, self.cells, self.T, self.steps = tiles, cells, T, steps
        self.obs = X[:, -steps:, 7] > 0.5
        self.age = X[:, -steps:, 8].astype(np.float32)

    def __len__(self):
        return len(self.cells)

    def get(self, i: np.ndarray, device: str):
        t = window_tiles(self.tiles, self.cells[i], self.T[i], self.steps)
        if self.steps == 1:
            t = t[:, 0]
        tt = torch.from_numpy(np.ascontiguousarray(t)).to(device)
        obs = torch.from_numpy(self.obs[i]).to(device)
        age = torch.from_numpy(self.age[i]).to(device)
        return tt, obs, age


def _forward(model, tt, obs, age, steps):
    if steps == 1:
        return model(to_input(tt))
    return model(to_input(tt), obs.float(), age)


def train(cfg: dict, fit: Batches, Y: np.ndarray, M: np.ndarray, seed: int, device: str = "mps", threads: int = 2,
          evals: list | None = None) -> dict:
    """One run, as nets.train: BCE (masked per head), AdamW, cosine decay to 5%, clip 1.0, fixed epochs; masking
    augmentation (cnn_bilstm) hides observed months T-11..T-1 with probability mask_p."""
    nets.setup_torch(seed, device, threads)
    rng = np.random.default_rng(seed)
    model = build(cfg, Y.shape[1]).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"])
    lossf = nn.BCEWithLogitsLoss(reduction="none")
    Yt = torch.from_numpy(Y.astype(np.float32)).to(device)
    Mt = torch.from_numpy(M.astype(np.float32)).to(device)
    steps, n, batch, epochs = fit.steps, len(fit), int(cfg.get("batch", 512)), int(cfg["epochs"])
    p = float(cfg.get("mask_p", 0.0)) if steps > 1 else 0.0
    gen = torch.Generator(device="cpu").manual_seed(seed)
    history, t_start = [], time.perf_counter()
    for epoch in range(1, epochs + 1):
        t0 = time.perf_counter()
        rate = nets._lr_at(epoch, epochs, cfg["lr"])
        for g in opt.param_groups:
            g["lr"] = rate
        model.train()
        order = rng.permutation(n)
        total, seen = 0.0, 0.0
        for s in range(0, n, batch):
            bi = order[s:s + batch]
            tt, obs, age = fit.get(bi, device)
            if p > 0:
                u = torch.rand(obs.shape, generator=gen).to(device)
                hide = (u < p) & obs
                hide[:, -1] = False                          # month T is never hidden (job 1)
                if bool(hide.any()):
                    tt = torch.where(hide[:, :, None, None], torch.full_like(tt, EMPTY), tt)
                    obs = obs & ~hide
                    age = age_from_obs(obs)
            yb, mb = Yt[bi], Mt[bi]
            opt.zero_grad(set_to_none=True)
            z = _forward(model, tt, obs, age, steps)
            loss = (lossf(z, yb) * mb).sum() / mb.sum().clamp(min=1.0)
            if not torch.isfinite(loss):
                raise RuntimeError("non-finite training loss")
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            w = float(mb.sum())
            total += float(loss.detach()) * w
            seen += w
        history.append({"epoch": epoch, "lr": rate, "train_loss": total / max(seen, 1),
                        "seconds": round(time.perf_counter() - t0, 2)})
    preds = {name: predict(model, b, device) for name, b in (evals or [])}
    return {"model": model, "history": history, "seconds": round(time.perf_counter() - t_start, 1),
            "n_params": nets.count_parameters(model), "logits": preds,
            "state": {k: v.detach().cpu() for k, v in model.state_dict().items()}}


@torch.inference_mode()
def predict(model, b: Batches, device: str, batch: int = 2048) -> np.ndarray:
    model.eval()
    out = []
    for s in range(0, len(b), batch):
        i = np.arange(s, min(s + batch, len(b)))
        tt, obs, age = b.get(i, device)
        out.append(_forward(model, tt, obs, age, b.steps).float().cpu().numpy())
    return np.concatenate(out) if out else np.zeros((0, 1), np.float32)
