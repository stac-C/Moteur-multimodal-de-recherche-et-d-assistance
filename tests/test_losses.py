import numpy as np
import pytest

torch = pytest.importorskip("torch")
from mmsearch.training.losses import info_nce, info_nce_tau  # noqa: E402

# Exemple chiffré de la section 1.5 : N = 2, d = 2, tau = 0,1, avec cos 30° arrondi à 0,8660
# (avec la valeur exacte 0,8660254, la perte vaut 0,025401 au lieu de 0,025408).
X = [[1.0, 0.0], [0.0, 1.0]]
Y = [[0.8660, 0.5], [0.5, 0.8660]]


def test_info_nce_value():
    loss = info_nce_tau(torch.tensor(X), torch.tensor(Y), 0.1)
    assert loss.item() == pytest.approx(0.025408, abs=2e-6)


def test_info_nce_gradient_matches_derivation():
    zx = torch.tensor(X, requires_grad=True)
    info_nce_tau(zx, torch.tensor(Y), 0.1).backward()
    assert zx.grad[0].numpy() == pytest.approx(np.array([-0.04591, 0.04591]), abs=2e-5)


def test_info_nce_logit_scale_equivalence():
    s = torch.tensor(float(np.log(1 / 0.1)))
    assert info_nce(torch.tensor(X), torch.tensor(Y), s).item() == pytest.approx(0.025408, abs=2e-6)
