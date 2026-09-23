# Models

The teacher and the student are the same architecture — `DroughtBiLSTM` in
[`src/dewsat/model.py`](../src/dewsat/model.py) — at two sizes. The teacher
learns from the labels. The student learns from the labels and from the
teacher's predictions (knowledge distillation, `distill_alpha` 0.4), which is
what `dewsat distill` does. Both have an INT8 copy, and the student is exported
for the Jetson.

```
models/
├── SHA256SUMS    a checksum for every file below
├── teacher/
│   ├── fp32/     the training run: best.pt, config, metrics, audit, REPORT.md, test predictions
│   └── int8/     dynamic-INT8 copy of fp32/best.pt, and quantization.json with its accuracy and size cost
└── student/
    ├── fp32/     the distillation run, same layout as the teacher's
    ├── int8/     dynamic-INT8 student, and quantization.json
    └── export/   the Jetson bundle: model.onnx, model_torchscript.pt, calibration tensor, deploy_manifest.json
```

**These are the original model files, unchanged.** Each is a byte-for-byte copy
of the file that produced the numbers in the documentation (where each came from
is under [Provenance](#provenance)), kept as it was so that later results can be
diagnosed against exactly these models. Confirm nothing has changed with:

```bash
make verify-models        # or: cd models && shasum -a 256 -c SHA256SUMS   (sha256sum -c on Linux)
```

They were trained on the synthetic table in [`data/`](../data/). They show that
the software works end to end and carry no drought skill, and the code will not
run them against a real-data manifest — train your own on HLS/FLDAS data
([below](#training-on-your-own-hlsfldas-data)).

## At a glance

| | Teacher | Student |
|---|---|---|
| Architecture | Bi-LSTM, 3 layers × 256 units per direction, 128-unit head | Bi-LSTM, 1 layer × 24 units per direction, 32-unit head |
| Parameters | 3,832,065 | 9,857 (389× fewer) |
| Config that reproduces it | [`configs/teacher.json`](../configs/teacher.json) | [`configs/student_24.json`](../configs/student_24.json) — see the note below |
| Trained with | `dewsat train` | `dewsat distill`, with the teacher above |
| Task | Soil-moisture percentile 3 months ahead, from a 12-month window | Same |
| Test RMSE, `fp32/` | 0.17456 | 0.17242 |
| Dry-month CSI, `fp32/` | 0.546 | 0.531 |
| Test RMSE, float → `int8/` | 0.17456 → 0.17431 | 0.17634 → 0.17611 † |
| Weights, float → INT8 | 15.33 MB → 3.83 MB | 39.4 kB → 9.9 kB |
| File, float → INT8 | 15.34 MB → 3.97 MB | 46.9 kB → 49.7 kB |
| CPU latency, batch 1 | 1.31 ms float, 2.56 ms INT8 | 0.140 ms float, **0.021 ms ONNX Runtime**, 0.344 ms INT8 † |

† Measured on the student's `int8/` and `export/`, which are an earlier training
of the same configuration than `fp32/best.pt` — see [Provenance](#provenance).

On the same 619 held-out windows, persistence scores 0.2930, seasonal
climatology 0.2426 and ridge regression 0.1711. Latency is from
[`results/benchmarks/`](../results/benchmarks/), measured on 10 September 2026 on
an Apple Silicon laptop CPU — not on a Jetson.

Before quoting any of this:

- **The student's config.** `configs/student.json` has since been changed to the
  paper's 64-unit size. The committed student is the 24-unit model; its
  exact config is `configs/student_24.json`, identical to `student/fp32/config.json`.
  Its `audit.json` records `--config configs/student.json` because that file
  said 24 units at the time.
- **INT8 is a size result, not a speed result.** Dynamic INT8 is about 2.5×
  slower than float for the student on this CPU, and the student's INT8 *file*
  is larger than its float one because container overhead outweighs the weight
  saving at this size. ONNX Runtime is what makes it fast.
- **INT8 scores depend on the batch size.** Dynamic INT8 picks its activation
  scales per batch, so a window's prediction shifts slightly with what else is
  in the batch. The INT8 scores above used batches of 512, and the files
  reproduce them exactly that way; at batch 1 the teacher scores 0.17452 and
  the student 0.17618. Expect that difference from a batch-1 on-device run.
- **Ridge regression matches both models on RMSE.** The Bi-LSTM's case is
  dry-event detection; see [docs/STATUS.md §2](../docs/STATUS.md#2-what-the-numbers-actually-say).

## Provenance

| Folder | Copied from | Written | Produced by |
|---|---|---|---|
| `teacher/fp32/` | `runs/teacher/` | 2026-09-10 22:13 | `dewsat train --config configs/teacher.json` |
| `teacher/int8/` | `deploy/teacher_int8/` | 2026-09-10 22:15 | `dewsat quantize --checkpoint runs/teacher/best.pt` |
| `student/fp32/` | `runs/student_distilled/` | 2026-09-10 22:13 | `dewsat distill --teacher runs/teacher/best.pt`, 24-unit config |
| `student/int8/` | `deploy/student_int8/` | 2026-09-10 19:09 | `dewsat quantize` on an earlier `runs/student_distilled/best.pt` |
| `student/export/` | `deploy/student/`, previously committed as `examples/deploy_student/` | 2026-09-10 19:09 | `dewsat export` on that same earlier checkpoint |

Paths written inside the files — the command in each `audit.json`, the student
checkpoint's reference to its teacher — are where each file was produced, before
this folder existed. They are deliberately left unedited.

**The student's `int8/` and `export/` do not hold the same weights as
`student/fp32/best.pt`.** They were made at 19:09 from an earlier training of
the same 24-unit configuration. At 22:13 the student was retrained against the
current teacher, and that earlier checkpoint was overwritten; it no longer
exists anywhere ([docs/STATUS.md §3](../docs/STATUS.md#3-what-is-not-done) records
this). The architecture and the input scaler are identical, so the input recipe
in `export/deploy_manifest.json` is correct for both. The weights are not: on
the 512 calibration windows the export's predictions are up to 0.119 from
`fp32/best.pt`'s, 0.019 on average. The earlier checkpoint's own scores survive
in `int8/quantization.json` (float test RMSE 0.17634) and
`export/deploy_manifest.json` (validation RMSE 0.15597). **When diagnosing an
on-device result, compare it against `export/`, not against
`fp32/test_predictions.csv`.**

Everything else reproduces exactly. Retraining with `configs/teacher.json`, then
distilling with `configs/student_24.json` from that teacher, gives weights
identical to `teacher/fp32/best.pt` and `student/fp32/best.pt` on the software in
`requirements-tested.txt` (checked 2026-09-23). A consistent INT8 copy and
export of `student/fp32/best.pt` can be made with README Steps 9–10; they go to
`deploy/`, and this folder stays as it is.

## Loading them

The float checkpoints carry their scaler, data manifest and training config, so
`dewsat predict` can go straight from a monthly CSV to predictions:

```bash
.venv/bin/dewsat predict --checkpoint models/student/fp32/best.pt \
    --data data/synthetic_monthly.csv --target-month 2024-12 --out runs/student_2024-12.csv
```

From Python:

```python
import torch
from dewsat.train import load_checkpoint

model, checkpoint = load_checkpoint("models/student/fp32/best.pt")   # or models/teacher/fp32/best.pt

torch.backends.quantized.engine = "qnnpack"      # the engine the INT8 files were quantized with
int8 = torch.jit.load("models/student/int8/model_int8_dynamic.pt")
with torch.inference_mode():
    prediction = int8(x)                         # x: float32 tensor [batch, 12, 8]

import onnxruntime
session = onnxruntime.InferenceSession("models/student/export/model.onnx")
prediction = session.run(None, {"window": x.numpy()})[0]
```

The INT8, ONNX and TorchScript files take windows that are already built and
scaled. `export/deploy_manifest.json` specifies how: the channel order, the
scaler statistics, the missing-month rule and a SHA-256 for each model file.
`export/calibration.npz` holds 512 ready-made training windows to try them on.
Jetson steps are in [docs/JETSON.md](../docs/JETSON.md).

## Training on your own HLS/FLDAS data

Train a new teacher and distill a new student on your own table, keeping the
configs and changing only the dates. The full sequence — teacher, student, INT8
for both, and the Jetson export — is in
[docs/REAL_DATA.md §3](../docs/REAL_DATA.md#3-running-it). It was run end to end
on a stand-in real-data table (300 cells, 2013–2025) with no code changes.
