"""submission/build.py --notes: the package README must carry our disclosures.

The organisers follow the README. Fine-tuning from a public checkpoint and rerunning after an
infrastructure crash are allowed only if disclosed, so the notes must land inside the zip.
"""
from __future__ import annotations

import json
import subprocess
import sys
import zipfile
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]


def _package(tmp_path: Path, *extra: str) -> str:
    j = tmp_path / "rollouts" / "json"; j.mkdir(parents=True)
    for i in range(301, 321):
        (j / f"turning_on_radio_{i}_0.json").write_text(json.dumps(
            {"task": "turning_on_radio", "instance_id": i, "rollout_id": 0, "q_score": {"final": 0.0}}))
    (tmp_path / "w.py").write_text("# wrapper\n"); (tmp_path / "r.yaml").write_text("robot: r1pro\n")
    out = tmp_path / "pkg.zip"
    p = subprocess.run([sys.executable, "-m", "submission.build", "--rollouts", str(tmp_path / "rollouts"),
                        "--wrapper", str(tmp_path / "w.py"), "--robot-config", str(tmp_path / "r.yaml"),
                        "--team", "t", "--affiliation", "a", "--date", "20261016",
                        "--intended-task", "turning_on_radio", "--out", str(out), *extra],
                       cwd=REPO, capture_output=True, text=True, timeout=120)
    assert p.returncode == 0, p.stdout + p.stderr
    with zipfile.ZipFile(out) as z:
        return z.read(next(n for n in z.namelist() if n.endswith("README.md"))).decode()


def test_notes_are_written_into_the_package_readme(tmp_path):
    notes = tmp_path / "notes.md"
    notes.write_text("Fine-tuned from the organisers' released checkpoint.\n\nRelaunched once after a driver crash.\n")
    readme = _package(tmp_path, "--notes", str(notes))
    assert "## Notes and disclosures" in readme
    assert "Fine-tuned from the organisers' released checkpoint." in readme
    assert "Relaunched once after a driver crash." in readme


def test_without_notes_the_readme_has_no_empty_section(tmp_path):
    assert "## Notes and disclosures" not in _package(tmp_path)
