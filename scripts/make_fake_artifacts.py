"""python scripts/make_fake_artifacts.py --n-img 200 --n-aud 100 : artefacts factices + index + SQLite."""

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from mmsearch.config import load_config  # noqa: E402
from mmsearch.fake import make_fake_artifacts  # noqa: E402
from mmsearch.index.build_index import build_all  # noqa: E402

if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--n-img", type=int, default=200)
    ap.add_argument("--n-aud", type=int, default=100)
    ap.add_argument("--n-esc", type=int, default=50)
    ap.add_argument("--root", default=str(ROOT))
    a = ap.parse_args()
    root = make_fake_artifacts(Path(a.root), a.n_img, a.n_aud, a.n_esc)
    print(build_all(load_config(root / "config.yaml")))
