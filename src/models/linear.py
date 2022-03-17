import jax.numpy as jnp

from src.models.model import State, Control


class LinearTimeInvariantModel:

    def __init__(self, A: jnp.ndarray, B: jnp.ndarray):
        self.A = A
        self.B = B

    def __call__(self, x: State, u: Control) -> State:
        return self.A @ x + self.B @ u