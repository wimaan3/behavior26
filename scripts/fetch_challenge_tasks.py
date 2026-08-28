"""Scrape the authoritative list of the 100 challenge tasks.

There is no task list in the BEHAVIOR-1K repo. OmniGibson reads the names from
    <gm.DATA_PATH>/2026-challenge-task-instances/metadata/B100_task_misc.csv
(see omnigibson/eval/utils/eval_utils.py::_load_challenge_task_misc), which is
part of the downloaded dataset, and the HuggingFace dataset repo is gated.

The public demo gallery at behavior.stanford.edu/challenge/tasks/index.html
carries the same 100 tasks inline as a JavaScript array, along with each task's
rooms, scene model and mean human demo duration. That is what this scrapes.

Cross-checks that should hold if the scrape is still valid:
  * exactly 100 tasks
  * 7 distinct scene models
  * mean duration ~350s (the published mean human demo length is 351s)

Usage:
    python scripts/fetch_challenge_tasks.py --out data/challenge_tasks.json
"""

import argparse
import json
import subprocess
import sys

URL = "https://behavior.stanford.edu/challenge/tasks/index.html"
MARKER = "const tasks = ["


def fetch(url):
    # NOTE: python's urllib and plain curl both get the connection reset by
    # this host; forcing HTTP/1.1 is what makes it work. If this breaks, open
    # the page in a browser and pull the `const tasks = [...]` array by hand.
    out = subprocess.run(
        ["curl", "-sSL", "--http1.1", "--max-time", "60", url],
        capture_output=True, check=True,
    )
    return out.stdout.decode("utf-8", errors="replace")


def extract(html):
    start = html.index(MARKER) + len(MARKER) - 1
    depth = 0
    for i in range(start, len(html)):
        if html[i] == "[":
            depth += 1
        elif html[i] == "]":
            depth -= 1
            if depth == 0:
                return json.loads(html[start:i + 1])
    raise ValueError("unterminated tasks array")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--url", default=URL)
    args = ap.parse_args()

    tasks = extract(fetch(args.url))

    scenes = sorted({t["scene_model"] for t in tasks})
    mean_duration = sum(t["duration"] for t in tasks) / len(tasks)
    print("tasks: %d" % len(tasks))
    print("scenes: %d %s" % (len(scenes), scenes))
    print("mean human demo duration: %.1fs" % mean_duration)
    if len(tasks) != 100:
        print("WARNING: expected 100 tasks, got %d" % len(tasks), file=sys.stderr)

    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(tasks, f, indent=1)
    print("wrote %s" % args.out)


if __name__ == "__main__":
    main()
