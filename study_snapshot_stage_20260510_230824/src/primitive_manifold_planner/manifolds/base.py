from __future__ import annotations

from abc import ABC, abstractmethod
import numpy as np

class ImplicitManifold(ABC):
    
    def __init__(self, ambient_dim: int, codim: int, name: str = "manifold") -> None:
        self.ambient_dim = ambient_dim
        self.codim = codim
        self.name = name
    @abstractmethod
    def residual(self, x: np.ndarray) -> np.ndarray:
        raise NotImplementedError
    
    @abstractmethod
    def jacobian(self, x: np.ndarray) -> np.ndarray:
        raise NotImplementedError
    
    def is_valid(self, x:np.ndarray, tol: float = 1e-6) -> bool:
        x = self._coerce_point(x)
        r = self.residual(x)
        return np.linalg.norm(r) <= tol
    def tangent_projector(self, x:np.ndarray) -> np.ndarray:
        x = self._coerce_point(x)
        j = self.jacobian(x)
        i = np.eye(self.ambient_dim)
        jj_t_inv = np.linalg.pinv(j @ j.T)
        return i - j.T @ jj_t_inv @ j
    def project_tangent(self, x: np.ndarray, v: np.ndarray) -> np.ndarray:
        x = self._coerce_point(x)
        v = self._coerce_vector(v)
        p = self.tangent_projector(x)
        return p @ v
    def _coerce_point(self, x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype = float).reshape(-1)
        if x.shape[0] != self.ambient_dim:
            raise ValueError(f"Point dimension mismatch: expected {self.ambient_dim}, got {x.shape[0]}")
        return x
    def _coerce_vector(self, v: np.ndarray) -> np.ndarray:
        v = np.asarray(v, dtype = float).reshape(-1)
        if v.shape[0] != self.ambient_dim:
            raise ValueError(f"Vector dimension mismatch: expected {self.ambient_dim}, got {v.shape[0]}")
        return v
    def __repr__(self):
        return (
            f"{self.__class__.__name__}("
            f"name='{self.name}', ambient_dim={self.ambient_dim}, codim={self.codim})"
        )
