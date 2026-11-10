# Source from the repository root before running anything: keeps every temp file and cache inside the repository.
export V2="$(pwd)"
export DEWSAT_V2="$V2"
[ -d "$V2/code/dewsat2" ] || echo "env.sh: source this file from the repository root" >&2
mkdir -p "$V2/tmp"
export TMPDIR="$V2/tmp"
export PIP_CACHE_DIR="$V2/pip-cache"
export MPLCONFIGDIR="$V2/mplconfig"
export XDG_CACHE_HOME="$V2/cache/xdg"
export PYTHONDONTWRITEBYTECODE=1
export PYTHONPYCACHEPREFIX="$V2/cache/pycache"
export GDAL_CACHEMAX=256
export CPL_TMPDIR="$V2/tmp"
export TORCH_HOME="$V2/cache/torch"
export HF_HOME="$V2/cache/hf"
export PATH="$V2/.venv/bin:$PATH"
