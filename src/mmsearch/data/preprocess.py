"""Prétraitement texte, image et audio (étape 2 du pipeline hors-ligne, section 2.2)."""

from __future__ import annotations

import re
import unicodedata
from pathlib import Path

import numpy as np

_CTRL = re.compile(r"[\u0000-\u0008\u000b\u000c\u000e-\u001f\u007f]")
_SPACES = re.compile(r"\s+")


def clean_text(text: str) -> str:
    """NFKC, suppression des caractères de contrôle, espaces normalisés.

    Pas de mise en minuscules ni de retrait de mots vides : les tokeniseurs
    Transformer (CLIP BPE, WordPiece) s'en chargent ou n'en ont pas besoin.
    """
    text = unicodedata.normalize("NFKC", str(text))
    text = _CTRL.sub(" ", text)
    return _SPACES.sub(" ", text).strip()


def load_image(path: str | Path):
    """Ouvre une image et la convertit en RGB (le redimensionnement 224x224 est fait par CLIP)."""
    from PIL import Image

    with Image.open(path) as im:
        return im.convert("RGB")


def load_audio(
    path: str | Path, sample_rate: int = 16000, max_seconds: float | None = None
) -> np.ndarray:
    """Charge un fichier audio en mono float32 à `sample_rate` Hz."""
    import librosa

    wave, _ = librosa.load(str(path), sr=sample_rate, mono=True, duration=max_seconds)
    return np.asarray(wave, dtype=np.float32)


def split_windows(
    wave: np.ndarray, sample_rate: int = 16000, window_s: float = 10.0, min_tail_s: float = 1.0
) -> list[np.ndarray]:
    """Découpe en fenêtres de `window_s` secondes sans recouvrement.

    Une dernière fenêtre plus courte que `min_tail_s` est ignorée, sauf si c'est la seule.
    Un signal vide devient une fenêtre de silence d'une seconde.
    """
    if wave.size == 0:
        return [np.zeros(sample_rate, dtype=np.float32)]
    step = int(window_s * sample_rate)
    chunks = [wave[i : i + step] for i in range(0, wave.size, step)]
    if len(chunks) > 1 and chunks[-1].size < int(min_tail_s * sample_rate):
        chunks = chunks[:-1]
    return chunks
