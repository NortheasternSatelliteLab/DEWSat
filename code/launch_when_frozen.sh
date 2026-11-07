#!/bin/zsh
# Wait until no further new-region batch will be pulled, freeze the region set, then start the pipeline through the
# development stages (Phase 5 and 6 are launched separately after a failure check). Run detached:
#   caffeinate -dimsu nohup code/launch_when_frozen.sh > logs/launcher.log 2>&1 &
source "$HOME/Desktop/DEWSat_BiLSTM_v2/env.sh"
cd "$V2"
while true; do
  if .venv/bin/python -I code/pull/freeze_regions.py > logs/freeze_check.log 2>&1; then
    echo "$(date '+%Y-%m-%d %H:%M:%S') region set frozen"; cat logs/freeze_check.log; break
  fi
  sleep 120
done
echo "$(date '+%Y-%m-%d %H:%M:%S') launching the pipeline (features .. weekly)"
exec .venv/bin/python -u code/dewsat2/run.py --stages features,simple,search,final,score,checks,whatif,weekly \
  --cpu-workers 4 --mps-workers 2 --threads 2 --feature-workers 6 >> logs/run.log 2>&1
