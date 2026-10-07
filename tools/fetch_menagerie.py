"""Fetch only the Stretch 3 folder of MuJoCo Menagerie into third_party/.

A shallow, sparse checkout at a pinned commit, so the exact-geometry tests always
compare against the same model the chain constants were extracted from.

    python tools/fetch_menagerie.py
"""

import subprocess
import sys
from pathlib import Path

REPO = "https://github.com/google-deepmind/mujoco_menagerie"
COMMIT = "f054586a8e90465d49ee5be15335c4a0c7f57caf"
TARGET = Path(__file__).resolve().parents[1] / "third_party" / "mujoco_menagerie"


def git(*args: str) -> None:
    subprocess.run(["git", "-C", str(TARGET), *args], check=True)


def main() -> None:
    if (TARGET / "hello_robot_stretch_3" / "stretch.xml").exists():
        print(f"Already present: {TARGET}")
        return
    TARGET.mkdir(parents=True, exist_ok=True)
    git("init", "-q")
    git("remote", "add", "origin", REPO)
    git("sparse-checkout", "set", "hello_robot_stretch_3")
    git("fetch", "-q", "--depth", "1", "--filter=blob:none", "origin", COMMIT)
    git("checkout", "-q", "FETCH_HEAD")
    print(f"Fetched hello_robot_stretch_3 @ {COMMIT[:10]} into {TARGET}")


if __name__ == "__main__":
    try:
        main()
    except subprocess.CalledProcessError as exc:
        sys.exit(f"git failed: {exc}")
