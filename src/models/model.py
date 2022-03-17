from typing import Protocol
import jax.numpy as jnp

State = jnp.ndarray
Control = jnp.ndarray
Observation = jnp.ndarray

class ForwardModel(Protocol):

    def __call__(self, x: State, u: Control) -> State:
        pass

class ObservationModel(Protocol):

    def __call__(self, x: State) -> Observation:
        pass