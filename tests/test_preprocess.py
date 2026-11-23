import numpy as np

from mmsearch.data.preprocess import clean_text, split_windows

SR = 16000


def test_clean_text():
    assert clean_text("  A\u00a0dog \t runs\n fast ") == "A dog runs fast"
    assert clean_text("ﬁne") == "fine"  # NFKC : ligature -> lettres


def test_split_windows():
    assert len(split_windows(np.zeros(25 * SR, dtype=np.float32), SR)) == 3  # 10 + 10 + 5 s
    assert (
        len(split_windows(np.zeros(int(10.5 * SR), dtype=np.float32), SR)) == 1
    )  # queue < 1 s ignorée
    assert len(split_windows(np.zeros(5 * SR, dtype=np.float32), SR)) == 1
    assert split_windows(np.zeros(0, dtype=np.float32), SR)[0].size == SR  # silence d'1 s
