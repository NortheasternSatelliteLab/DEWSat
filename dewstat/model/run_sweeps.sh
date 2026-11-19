#!/usr/bin/env bash
# Train V1, V2 and V3 on all regions: 5 settings x 3 seeds each. The setting to use is picked afterwards by
# validation loss (make_results.py), never by test score.
#     bash run_sweeps.sh            # HLS 2020+ only (pc), the chosen setup
#     bash run_sweeps.sh pc full    # also HLS 2016+ (needs the "nasa" pull step and merge_hls_years.py)
# Results: runs_<setup>_<inputs>/<name>/ (summary.txt, scores.json, plots/). ~3 min per setting on 6 CPU cores.
set -uo pipefail
cd "$(dirname "$0")"
py=.venv-train/bin/python
[ -x "$py" ] || py=.venv-train/Scripts/python.exe  # Windows (Git Bash)
setups=("${@:-pc}")
configs=(
    "h16_win6:--hidden 16 --window 6"
    "h16_win3:--hidden 16 --window 3"
    "h16_win12:--hidden 16 --window 12"
    "h32_win6:--hidden 32 --window 6"
    "h16_win6_noloc:--hidden 16 --window 6 --no-location"
)
for hls in "${setups[@]}"; do
    for job in "satellite_anomaly:anom" "satellite_swir:swir" "satellite:sat"; do
        IFS=: read -r inputs prefix <<< "$job"
        runs_dir="runs_${hls}_${inputs}"
        mkdir -p "$runs_dir"
        for c in "${configs[@]}"; do
            name="${prefix}_${c%%:*}"
            [ -f "$runs_dir/$name/scores.json" ] && { echo "skip $runs_dir/$name (done)"; continue; }
            echo "$(date '+%H:%M:%S') train $runs_dir/$name"
            # shellcheck disable=SC2086
            $py train.py --inputs "$inputs" --hls "$hls" --runs-dir "$runs_dir" --dropout 0.5 --weight-decay 1e-2 \
                --name "$name" --seeds 3 --quiet ${c#*:} > "$runs_dir/$name.log" 2>&1 \
                || echo "FAILED $runs_dir/$name (see $runs_dir/$name.log)"
        done
        $py plot_results.py --compare --runs-dir "$runs_dir" > /dev/null 2>&1 || true
    done
done
$py make_results.py
