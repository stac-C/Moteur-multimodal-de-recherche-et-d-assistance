"""Fixtures partagées : artefacts factices, index, moteur avec encodeurs simulés (sans torch)."""

import io

import numpy as np
import pytest
from PIL import Image

from mmsearch.config import load_config
from mmsearch.data.loaders import load_captions, load_manifest
from mmsearch.embeddings_io import load_embeddings
from mmsearch.fake import make_fake_artifacts
from mmsearch.index.build_index import build_all
from mmsearch.retrieval.engine import SearchEngine


class StubText:
    """Texte -> embedding de la légende correspondante (sinon vecteur pseudo-aléatoire stable)."""

    def __init__(self, cfg):
        caps = load_captions(cfg).set_index("caption_id")["text"].to_dict()
        self.vec = {}
        for f in ("clip_txt_flickr8k", "clip_txt_clotho"):
            emb, ids = load_embeddings(cfg.emb_dir / f"{f}.npy")
            self.vec.update({caps[i]: emb[n] for n, i in enumerate(ids)})
        man = load_manifest(cfg)
        esc = man[man["dataset"] == "esc50"]
        a, ids = load_embeddings(cfg.emb_dir / "audio_esc50.npy")
        lab = esc.set_index("item_id").loc[ids, "label"].to_numpy()
        for c in sorted(set(lab)):
            v = a[lab == c].mean(axis=0)
            self.vec[f"the sound of {c.replace('_', ' ')}"] = v / np.linalg.norm(v)

    def encode(self, texts, batch_size=128):
        out = []
        for t in texts:
            if t not in self.vec:
                r = np.random.default_rng(abs(hash(t)) % 2**32).normal(size=512)
                self.vec[t] = (r / np.linalg.norm(r)).astype(np.float32)
            out.append(self.vec[t])
        return np.stack(out).astype(np.float32)


class StubImage:
    """Pixel (0, 0) = (indice mod 256, indice // 256, 0) -> embedding de l'image d'indice donné."""

    def __init__(self, cfg):
        self.emb, _ = load_embeddings(cfg.emb_dir / "clip_img_flickr8k.npy")

    def encode(self, inputs, batch_size=64):
        return np.stack(
            [self.emb[(lambda p: p[0] + 256 * p[1])(x.getpixel((0, 0)))] for x in inputs]
        )


class StubAudio:
    """Le fichier contient un indice ASCII -> embedding du clip Clotho correspondant."""

    def __init__(self, cfg):
        self.emb, _ = load_embeddings(cfg.emb_dir / "audio_clotho.npy")

    def encode(self, paths, batch_size=8):
        out = []
        for p in paths:
            txt = open(p, "rb").read().decode()
            if not txt.strip().isdigit():
                raise ValueError("format audio non reconnu")
            out.append(self.emb[int(txt)])
        return np.stack(out)


def png_bytes(index: int) -> bytes:
    buf = io.BytesIO()
    Image.new("RGB", (8, 8), (index % 256, index // 256, 0)).save(buf, format="PNG")
    return buf.getvalue()


@pytest.fixture(scope="session")
def cfg(tmp_path_factory):
    root = make_fake_artifacts(tmp_path_factory.mktemp("fake"), n_img=60, n_aud=40, n_esc=30)
    c = load_config(root / "config.yaml")
    build_all(c)
    return c


@pytest.fixture()
def engine(cfg):
    stubs = {
        "text": lambda: StubText(cfg),
        "image": lambda: StubImage(cfg),
        "audio": lambda: StubAudio(cfg),
    }
    return SearchEngine.from_artifacts(cfg, encoders=stubs)
