#!/usr/bin/env bash
# Pull demonstration data. START SMALL.
#
#   LeRobot demos : 3.27 TB   behavior-1k/2026-challenge-demos
#   Raw HDF5      : 1.44 TB   behavior-1k/2026-challenge-rawdata
#
# Tasks are chunks: chunk-000 is task 0. Pull a 10-task slice (~330 GB) first,
# measure read throughput, and only then decide on the full set (decision D3).
set -euo pipefail

DATA_ROOT="${DATA_ROOT:-$HOME/behavior-data}"
CHUNKS="${CHUNKS:-chunk-000}"

mkdir -p "${DATA_ROOT}"
echo "==> downloading ${CHUNKS} into ${DATA_ROOT}"

for CHUNK in ${CHUNKS}; do
  huggingface-cli download behavior-1k/2026-challenge-demos \
    --repo-type dataset --local-dir "${DATA_ROOT}" \
    --include "data/${CHUNK}/**" \
    --include "meta/episodes/${CHUNK}/**" \
    --include "videos/*/${CHUNK}/**" \
    --include "meta/info.json" \
    --include "meta/stats.json" \
    --include "meta/tasks.parquet" \
    --include "meta/tasks.jsonl"
done

du -sh "${DATA_ROOT}"

cat <<'NOTES'

==> two things to answer while you are in here (Week 1, questions Q7 and Q8)

    Q7  Does the data contain PER-TIMESTEP predicate state, or must we replay
        20,000 demos through the simulator to recover it?
        Biggest cost unknown in the plan -- an afternoon vs. two weeks.

    Q8  Inspect the skill annotations: 31 unique skills, 270,600 segments,
        ~27 per trajectory. If usable, that is a pre-labelled sub-goal
        decomposition for all 100 tasks, free, and a far denser progress
        signal than the 2-3 goal predicates most tasks have.

    Look in:  annotations/  and  meta/
NOTES
