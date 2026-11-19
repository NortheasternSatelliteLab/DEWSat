#!/usr/bin/env bash
# Weekly vs monthly, both with NLDAS-2 as the answer key (V2 inputs, same regions and splits).
# 5 settings x 3 seeds per step; the setting is picked by validation loss afterwards (make_results_nldas.py).
#     bash run_nldas.sh
# Needs data-pipeline/practice/nldas_daily.npz (nldas_daily.py, 2001-2025). Results: runs_nldas_{month,week}/<name>/
set -uo pipefail
cd "$(dirname "$0")"
py=.venv-train/bin/python
[ -x "$py" ] || py=.venv-train/Scripts/python.exe  # Windows (Git Bash)
declare -A configs=(
    [month]="h16_win3:--hidden 16 --window 3|h16_win6:--hidden 16 --window 6|h16_win12:--hidden 16 --window 12|h32_win6:--hidden 32 --window 6|h16_win6_noloc:--hidden 16 --window 6 --no-location"
    [week]="h16_win4:--hidden 16 --window 4|h16_win13:--hidden 16 --window 13|h16_win26:--hidden 16 --window 26|h32_win13:--hidden 32 --window 13|h16_win52:--hidden 16 --window 52"
)
for step in month week; do
    runs_dir="runs_nldas_${step}"
    mkdir -p "$runs_dir"
    IFS='|' read -ra cs <<< "${configs[$step]}"
    for c in "${cs[@]}"; do
        name="anom_${c%%:*}"
        [ -f "$runs_dir/$name/scores.json" ] && { echo "skip $runs_dir/$name (done)"; continue; }
        echo "$(date '+%H:%M:%S') train $runs_dir/$name"
        # shellcheck disable=SC2086
        $py train.py --answer "nldas_${step}" --inputs satellite_anomaly --runs-dir "$runs_dir" \
            --dropout 0.5 --weight-decay 1e-2 --name "$name" --seeds 3 --quiet ${c#*:} > "$runs_dir/$name.log" 2>&1 \
            || echo "FAILED $runs_dir/$name (see $runs_dir/$name.log)"
    done
done
$py make_results_nldas.py
echo "$(date '+%H:%M:%S') ALL DONE"
