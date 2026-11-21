import os
import sys

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from tools.generate_assets import generate  # noqa: E402


@pytest.fixture(scope="session")
def assets(tmp_path_factory):
    root = tmp_path_factory.mktemp("recording")
    assets_dir = str(root / "assets")
    rec_dir = str(root / "recordings")
    summary = generate(assets_dir, rec_dir)
    return {
        "vpc": os.path.join(assets_dir, "scene.vpc"),
        "wav": os.path.join(assets_dir, "drums.wav"),
        "beatmap": os.path.join(assets_dir, "beatmap.json"),
        "plan": os.path.join(rec_dir, "replay_plan.json"),
        "root": str(root),
        **summary,
    }
