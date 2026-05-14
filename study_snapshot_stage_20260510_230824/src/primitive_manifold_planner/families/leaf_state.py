from __future__ import annotations

from dataclasses import dataclass
import numpy as np


@dataclass
class LeafState:
    family_name: str
    lam: object
    x: np.ndarray

    def copy(self) -> "LeafState":
        return LeafState(
            family_name=self.family_name,
            lam=self.lam,
            x=self.x.copy(),
        )