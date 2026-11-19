#!/usr/bin/env bash
# Same sweeps as run_sweeps.sh (FLDAS monthly V2 / V1) and run_nldas.sh / run_nomap.sh (NLDAS weekly), but validation = val regions in 2023 only
# (--val-years gap), so validation shares no year with training (<= 2022) or test (2024+). Picked by val loss later.
#     bash run_valgap.sh > run_valgap.log 2>&1
set -uo pipefail
cd "$(dirname "$0")"
py=.venv-train/bin/python
month="h16_win6:--hidden 16 --window 6|h16_win3:--hidden 16 --window 3|h16_win12:--hidden 16 --window 12|h32_win6:--hidden 32 --window 6|h16_win6_noloc:--hidden 16 --window 6 --no-location"
week="h16_win4:--hidden 16 --window 4|h16_win13:--hidden 16 --window 13|h16_win26:--hidden 16 --window 26|h32_win13:--hidden 32 --window 13|h16_win52:--hidden 16 --window 52"
# runs_dir:answer:inputs:prefix:configs
for job in "runs_valgap_fldas:fldas:satellite_anomaly:anom:month" "runs_valgap_fldas_nomap:fldas:satellite:sat:month" \
           "runs_valgap_week:nldas_week:satellite_anomaly:anom:week" "runs_valgap_week_nomap:nldas_week:satellite:sat:week"; do
    IFS=: read -r runs_dir answer inputs prefix cfg <<< "$job"
    mkdir -p "$runs_dir"
    IFS='|' read -ra cs <<< "${!cfg}"
    for c in "${cs[@]}"; do
        name="${prefix}_${c%%:*}"
        [ -f "$runs_dir/$name/scores.json" ] && { echo "skip $runs_dir/$name (done)"; continue; }
        echo "$(date '+%H:%M:%S') train $runs_dir/$name"
        # shellcheck disable=SC2086
        $py train.py --answer "$answer" --inputs "$inputs" --hls pc --runs-dir "$runs_dir" \
            --val-years gap --dropout 0.5 --weight-decay 1e-2 --name "$name" --seeds 3 --quiet ${c#*:} \
            > "$runs_dir/$name.log" 2>&1 || echo "FAILED $runs_dir/$name (see $runs_dir/$name.log)"
    done
done
echo "$(date '+%H:%M:%S') ALL DONE"
