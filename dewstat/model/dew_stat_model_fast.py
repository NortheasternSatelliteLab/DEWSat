# Bi-LSTM, fast version.
#
# Copied from "dew_stat_model - Austin Schlather.py" (the original is unchanged).
# Same math, same inputs/outputs, but faster and ready for a real training run:
#
#   1. The hand-written gates are replaced by nn.LSTM(bidirectional=True).
#      It does exactly what lstm_unit() did (forget / input / limbo / output gates,
#      long_memory = f*long + i*l, short_memory = o*tanh(long)), but:
#        - all 4 gates are ONE matrix multiply instead of 8 small ones per step
#        - forward and backward run as one fused cuDNN kernel on the GPU
#        - no Python loop over time steps
#      PyTorch names: c = long_memory, h = short_memory, g = the "limbo" gate.
#   2. Dropout before the head (helps with ~30k samples; we overfit easily).
#   3. validation_step  -> lets training stop early and keep the best epoch.
#      predict_step     -> used to write predictions after training.
#   4. AdamW with a small weight decay + lower the learning rate when val loss stalls.
#   5. val_r2 / test_r2 logged every epoch next to the loss.

import torch
import torch.nn as nn
import lightning as L
from torchmetrics import R2Score


class BiLSTM(L.LightningModule):
    def __init__(self, input_size, hidden_size, lr=1e-3, num_layers=1, dropout=0.3, weight_decay=1e-4):
        super().__init__()
        self.save_hyperparameters()
        # Replaces the W / U / b ParameterDicts for "fwd" and "bwd".
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,  # input is (batch, seq_len, input_size), same as before
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.drop = nn.Dropout(dropout)
        # head: Linear(2*hidden_size, 1), same as before
        self.head = nn.Linear(2 * hidden_size, 1)
        # R^2 tracked over each whole epoch (not averaged per batch, which would be wrong for R^2)
        self.val_r2 = R2Score()
        self.test_r2 = R2Score()

    def forward(self, input):
        # h_n: final short_memory of every layer and direction, shape (layers*2, batch, hidden)
        _, (h_n, _) = self.lstm(input)
        short_fwd = h_n[-2]  # last layer, forward  (after reading t = 0 .. T-1)
        short_bwd = h_n[-1]  # last layer, backward (after reading t = T-1 .. 0)
        # Connecting them together
        connected = torch.cat([short_fwd, short_bwd], dim=1)
        return self.head(self.drop(connected))

    def configure_optimizers(self):
        opt = torch.optim.AdamW(self.parameters(), lr=self.hparams.lr, weight_decay=self.hparams.weight_decay)
        sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, factor=0.5, patience=5)
        return {"optimizer": opt, "lr_scheduler": {"scheduler": sched, "monitor": "val_loss"}}

    def _loss(self, batch):
        x, y = batch
        y_hat = self(x)
        return nn.functional.mse_loss(y_hat, y)

    def training_step(self, batch, batch_idx):
        loss = self._loss(batch)
        self.log("train_loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        self.log("val_loss", nn.functional.mse_loss(y_hat, y), on_epoch=True, prog_bar=True)
        # R^2 of the month-to-month change: 0 = no better than "same as last month", 1 = perfect
        self.val_r2.update(y_hat.flatten(), y.flatten())
        self.log("val_r2", self.val_r2, on_epoch=True, prog_bar=True)

    def test_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        self.log("test_loss", nn.functional.mse_loss(y_hat, y), on_epoch=True)
        self.test_r2.update(y_hat.flatten(), y.flatten())
        self.log("test_r2", self.test_r2, on_epoch=True)

    def predict_step(self, batch, batch_idx):
        x, _ = batch
        return self(x)
