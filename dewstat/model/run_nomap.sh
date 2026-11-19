#!/usr/bin/env bash
# Without the normal map: the model predicts the soil-moisture level itself (inputs "satellite", as V1), on the chosen
# HLS 2020+ data, for each answer key: FLDAS monthly, NLDAS monthly, NLDAS weekly. Same 5 settings x 3 seeds as the
# runs with the map; the setting is picked by validation loss afterwards (make_results_nomap.py).
#     bash run_nomap.sh
set -uo pipefail
cd "$(dirname "$0")"
py=.venv-train/bin/python
[ -x "$py" ] || py=.venv-train/Scripts/python.exe  # Windows (Git Bash)
month="h16_win6:--hidden 16 --window 6|h16_win3:--hidden 16 --window 3|h16_win12:--hidden 16 --window 12|h32_win6:--hidden 32 --window 6|h16_win6_noloc:--hidden 16 --window 6 --no-location"
week="h16_win4:--hidden 16 --window 4|h16_win13:--hidden 16 --window 13|h16_win26:--hidden 16 --window 26|h32_win13:--hidden 32 --window 13|h16_win52:--hidden 16 --window 52"
for job in "fldas:runs_pc_satellite:$month" "nldas_month:runs_nldas_month_nomap:$month" "nldas_week:runs_nldas_week_nomap:$week"; do
    answer=${job%%:*}; rest=${job#*:}; runs_dir=${rest%%:*}; configs=${rest#*:}
    mkdir -p "$runs_dir"
    IFS='|' read -ra cs <<< "$configs"
    for c in "${cs[@]}"; do
        name="sat_${c%%:*}"
        [ -f "$runs_dir/$name/scores.json" ] && { echo "skip $runs_dir/$name (done)"; continue; }
        echo "$(date '+%H:%M:%S') train $runs_dir/$name"
        # shellcheck disable=SC2086
        $py train.py --answer "$answer" --inputs satellite --hls pc --runs-dir "$runs_dir" \
            --dropout 0.5 --weight-decay 1e-2 --name "$name" --seeds 3 --quiet ${c#*:} > "$runs_dir/$name.log" 2>&1 \
            || echo "FAILED $runs_dir/$name (see $runs_dir/$name.log)"
    done
done
$py make_results_nomap.py
echo "$(date '+%H:%M:%S') ALL DONE"
