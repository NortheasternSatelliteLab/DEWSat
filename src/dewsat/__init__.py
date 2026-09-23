"""DEWSat v2: monthly satellite summaries -> Bi-LSTM -> a deployable, measured model.

Import order matters here. cuBLAS reads CUBLAS_WORKSPACE_CONFIG when it creates
its handle, so it has to be set before torch initialises CUDA. v1 set it inside
a function that ran after import, which is usually too late to have any effect.
"""
import os

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

from .contract import (CSV_FEATURES, MODEL_FEATURES, contract_fingerprint,  # noqa: E402
                       month_number, month_string, read_manifest)
from .data import (build_inputs, fit_scaler, make_windows, read_monthly_csv,  # noqa: E402
                   split_windows)
from .model import DroughtBiLSTM, distillation_loss, predict_batches, weighted_huber  # noqa: E402
from .synth import generate_synthetic  # noqa: E402
from .train import (DEFAULT_CONFIG, VERSION, load_checkpoint, load_config,  # noqa: E402
                    predict_csv, prepare, seed_everything, train)

__version__ = VERSION
__all__ = ["CSV_FEATURES", "MODEL_FEATURES", "DEFAULT_CONFIG", "VERSION", "DroughtBiLSTM",
           "build_inputs", "contract_fingerprint", "distillation_loss", "fit_scaler",
           "generate_synthetic", "load_checkpoint", "load_config", "make_windows",
           "month_number", "month_string", "predict_batches", "predict_csv", "prepare",
           "read_manifest", "read_monthly_csv", "seed_everything", "split_windows", "train",
           "weighted_huber"]
