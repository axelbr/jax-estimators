from typing import Protocol

import jax.numpy as jnp



class Estimator(Protocol):

    def __call__(self, x: jnp.ndarray, u: jnp.ndarray) -> jnp.ndarray:
        pass
