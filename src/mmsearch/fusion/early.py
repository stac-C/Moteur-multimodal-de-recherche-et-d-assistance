"""Fusion précoce (équation E6) : z_fus = norm(W_f [z_v ; z_c] + b_f), W_f dans R^{512x1024}."""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class EarlyFusion(nn.Module):
    """524 800 paramètres (524 288 + 512) + `logit_scale` utilisé à l'entraînement seulement.

    Initialisation = fusion tardive : W_f = [w_mod I | w_cap I], b_f = 0. L'entraînement part donc du
    score tardif (w_mod, w_cap) et ne peut que l'améliorer sur la validation (arrêt anticipé).
    """

    def __init__(
        self, dim: int = 512, w_mod: float = 0.7, w_cap: float = 0.3, init_tau: float = 0.07
    ) -> None:
        super().__init__()
        self.fc = nn.Linear(2 * dim, dim)
        with torch.no_grad():
            eye = torch.eye(dim)
            self.fc.weight.copy_(torch.cat([w_mod * eye, w_cap * eye], dim=1))
            self.fc.bias.zero_()
        self.logit_scale = nn.Parameter(torch.tensor(float(np.log(1.0 / init_tau))))

    def forward(self, z_mod: torch.Tensor, z_cap: torch.Tensor) -> torch.Tensor:
        return F.normalize(self.fc(torch.cat([z_mod, z_cap], dim=-1)), dim=-1)

    @torch.no_grad()
    def fuse_numpy(self, z_mod: np.ndarray, z_cap: np.ndarray, device: str = "cpu") -> np.ndarray:
        self.eval()
        a = torch.from_numpy(np.ascontiguousarray(z_mod)).to(device)
        b = torch.from_numpy(np.ascontiguousarray(z_cap)).to(device)
        return self(a, b).cpu().numpy().astype(np.float32)
