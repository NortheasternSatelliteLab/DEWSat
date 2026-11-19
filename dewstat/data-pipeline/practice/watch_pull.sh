#!/usr/bin/env bash
# Watch a running pull_regions.py: memory, and how often Planetary Computer / NASA push back.
# One line per interval: available RAM, swap used, total RAM of the build_table.py processes, and the
# number of NEW 403 / 429 / 5xx / connection-error retries written to regions/*/pull_log.txt since the last line.
# Run from this folder in a second tmux pane:   bash watch_pull.sh [seconds, default 60] | tee watch_log.txt
# Rule of thumb: available RAM under ~1 GB, swap growing, or retries climbing every interval -> stop the pull
# (Ctrl-C; it resumes from its caches) and restart it with a lower --parallel.
set -uo pipefail
cd "$(dirname "$0")"
every="${1:-60}"

count() {  # retry lines per status across all region logs: "403 429 5xx other"
    cat regions/*/pull_log.txt 2>/dev/null | grep -oE "(retry crop [^ ]+|: [0-9A-Za-z]+, retry)" | awk '
        { s = $0; sub(/^retry crop /, "", s); sub(/^: /, "", s); sub(/, retry$/, "", s)
          if (s == "403") a++; else if (s == "429") b++; else if (s ~ /^5[0-9][0-9]$/) c++; else d++ }
        END { printf "%d %d %d %d\n", a, b, c, d }'
}

read -r p403 p429 p5xx pother < <(count)
printf "%-8s %9s %8s %10s %5s %5s %5s %6s\n" time avail_RAM swap pull_RSS 403 429 5xx other
while true; do
    sleep "$every"
    read -r n403 n429 n5xx nother < <(count)
    avail=$(free -m | awk '/^Mem:/ {printf "%.1fG", $7 / 1024}')
    swap=$(free -m | awk '/^Swap:/ {printf "%.1fG", $3 / 1024}')
    rss=$(ps -eo rss,args | awk '/[b]uild_table.py/ {s += $1} END {printf "%.1fG", s / 1048576}')
    printf "%-8s %9s %8s %10s %5d %5d %5d %6d\n" "$(date +%H:%M:%S)" "$avail" "$swap" "$rss" \
        $((n403 - p403)) $((n429 - p429)) $((n5xx - p5xx)) $((nother - pother))
    p403=$n403 p429=$n429 p5xx=$n5xx pother=$nother
done
