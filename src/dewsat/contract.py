"""The data contract: CSV schema, month arithmetic, and manifest validation.

The CSV column schema is deliberately identical to dewsat-starter v1, so any
preprocessing already written against that schema transfers here unchanged.
What changed in v2 is what the *model* is allowed to do with the table:
missing months are now representable, and the target is read as "the value for
this row's month" so a forecast lead time becomes a modelling choice instead of
a property baked into the file.
"""
import hashlib
import json
import math
import re
from pathlib import Path

# Columns the CSV must provide, in the fixed order the model consumes them.
CSV_FEATURES = ["NDVI_mean", "NDVI_std", "VCI_mean", "VCI_p10",
                "valid_frac", "month_sin", "month_cos"]
# Appended by the loader, never read from the CSV: 1.0 observed, 0.0 imputed.
OBSERVED_FEATURE = "observed"
MODEL_FEATURES = [*CSV_FEATURES, OBSERVED_FEATURE]
# Indices into MODEL_FEATURES that get standardized. Quality, season and the
# observed flag are already on fixed, interpretable scales.
SCALED_INDICES = [0, 1, 2, 3]

SCHEMA_VERSIONS = (1, 2)
TARGET_TIMINGS = ("window_end_month", "value_for_its_own_month")
MANIFEST_REQUIRED = ("schema_version", "data_kind", "target_name", "target_range",
                     "target_timing", "spatial_unit", "feature_names", "preprocessing")
# Fields that define what the model learned. A change here invalidates a checkpoint.
CONTRACT_FIELDS = ("data_kind", "target_name", "target_range", "spatial_unit",
                   "feature_names", "preprocessing")

MONTH_PATTERN = re.compile(r"\d{4}-(0[1-9]|1[0-2])")


def month_number(value):
    """Map YYYY-MM to a dense integer so month arithmetic is plain subtraction."""
    if not MONTH_PATTERN.fullmatch(str(value)):
        raise ValueError(f"Month must be YYYY-MM, got {value!r}")
    year, month = map(int, str(value).split("-"))
    return year * 12 + month - 1


def month_string(number):
    return f"{number // 12:04d}-{number % 12 + 1:02d}"


def season_features(number):
    """sin/cos of the calendar month, matching the v1 formula exactly."""
    angle = 2 * math.pi * (number % 12) / 12
    return math.sin(angle), math.cos(angle)


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False, sort_keys=False) + "\n")


def read_json(path):
    return json.loads(Path(path).read_text())


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_object(value):
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode()).hexdigest()


def read_manifest(csv_path, manifest_path=None):
    """Load and check the sidecar manifest that states what the numbers mean."""
    path = Path(manifest_path) if manifest_path else Path(csv_path).with_suffix(".json")
    if not path.exists():
        raise FileNotFoundError(
            f"No manifest at {path}. Copy configs/real_manifest.template.json beside "
            "the CSV using the same base name, or pass --manifest.")
    manifest = read_json(path)
    missing = [key for key in MANIFEST_REQUIRED if key not in manifest]
    if missing:
        raise ValueError(f"Manifest missing required keys: {missing}")
    if manifest["schema_version"] not in SCHEMA_VERSIONS:
        raise ValueError(f"schema_version must be one of {list(SCHEMA_VERSIONS)}.")
    if manifest["feature_names"] != CSV_FEATURES:
        raise ValueError(f"feature_names must be exactly {CSV_FEATURES}.")
    if manifest["data_kind"] not in {"synthetic", "real"}:
        raise ValueError("data_kind must be 'synthetic' or 'real'.")
    if manifest["target_range"] != [0, 1]:
        raise ValueError("This model has a bounded head; target_range must be [0, 1].")
    if manifest["target_timing"] not in TARGET_TIMINGS:
        raise ValueError(f"target_timing must be one of {list(TARGET_TIMINGS)}.")
    if not manifest["preprocessing"] or "REPLACE_ME" in json.dumps(manifest):
        raise ValueError("Fill in every REPLACE_ME in the preprocessing manifest before training.")
    if manifest["data_kind"] == "real" and any(
            "synthetic" in str(manifest[key]).lower() for key in ("target_name", "spatial_unit")):
        raise ValueError("Real data must not be described with synthetic target or unit names.")
    return manifest


def contract_fingerprint(manifest):
    """Hash only the fields a trained model depends on.

    v1 compared whole manifests, so adding a processing date to an inference
    manifest was enough to make a valid checkpoint refuse to run. Dates, notes
    and provenance may drift; feature and target definitions may not.
    """
    return sha256_object({key: manifest.get(key) for key in CONTRACT_FIELDS})


def compare_manifests(training, inference):
    """Report which non-contract manifest fields drifted, for the audit trail."""
    drift = {}
    for key in sorted(set(training) | set(inference)):
        if key in CONTRACT_FIELDS or key == "schema_version":
            continue
        if training.get(key) != inference.get(key):
            drift[key] = {"training": training.get(key), "inference": inference.get(key)}
    return drift
