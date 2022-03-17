from dataclasses import dataclass
from typing import Dict, List
import jax.numpy as jnp
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation
import math

@dataclass
class History:
    states: List[jnp.ndarray]
    state_estimates: List[jnp.ndarray]
    covariance_matrices: List[jnp.ndarray]
    observations: List[jnp.ndarray]

    def __init__(self):
        self.states = []
        self.covariance_matrices = []
        self.observations = []
        self.state_estimates = []

    def update(self, x, z, x_hat, covariance):
        self.states.append(x)
        self.observations.append(z)
        self.state_estimates.append(x_hat)
        self.covariance_matrices.append(covariance)

def _plot_covariance_ellipse(x_hat, x_cov):  # pragma: no cover
    position_cov = x_cov[0:2, 0:2]
    eigval, eigvec = jnp.linalg.eig(position_cov)

    if eigval[0] >= eigval[1]:
        bigind = 0
        smallind = 1
    else:
        bigind = 1
        smallind = 0

    t = jnp.arange(0, 2 * jnp.pi + 0.1, 0.1)
    a = jnp.sqrt(eigval[bigind])
    b = jnp.sqrt(eigval[smallind])
    x = [a * jnp.cos(it) for it in t]
    y = [b * jnp.sin(it) for it in t]
    angle = math.atan2(eigvec[1, bigind].real, eigvec[0, bigind].real)
    rot = Rotation.from_euler('z', angle).as_matrix()[0:2, 0:2]
    fx = rot @ (jnp.array([x, y]))
    px = jnp.array(fx[0, :] + x_hat[0])
    py = jnp.array(fx[1, :] + x_hat[1])
    plt.plot(px, py, "--r")


def plot(data: History):
    observations = jnp.stack(data.observations)
    states = jnp.stack(data.states)
    state_estimates = jnp.stack(data.state_estimates)
    covariance_matrices = jnp.stack(data.covariance_matrices)

    plt.cla()
    plt.gcf().canvas.mpl_connect('key_release_event', lambda event: [exit(0) if event.key == 'escape' else None])
    plt.plot(observations[:, 0], observations[:, 1], c='orange')
    plt.plot(state_estimates[:, 0], state_estimates[:, 1], c='green')
    plt.plot(states[:, 0], states[:, 1])
    _plot_covariance_ellipse(state_estimates[-1], covariance_matrices[-1])

    plt.axis("equal")
    plt.grid(True)
    plt.pause(0.000001)