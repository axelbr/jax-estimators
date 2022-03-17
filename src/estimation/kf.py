from typing import Callable, Tuple

import jax.numpy as jnp
from jax import jacfwd


class ExtendedKalmanFilter:
    """
    Extended Kalman Filter for non-linear, non-additive noise process and observation models.
    x_{t+1} = f(x_t, u_t, w_t)
    z_t = h(x_t, v_t)
    """

    def __init__(self,
                 process_model: Callable[[jnp.ndarray, jnp.ndarray, jnp.ndarray], jnp.ndarray],
                 observation_model: Callable[[jnp.ndarray, jnp.ndarray], jnp.ndarray],
                 process_noise_covariance: jnp.ndarray,
                 observation_noise_covariance: jnp.ndarray):
        self.f = process_model
        self.h = observation_model
        self.F = jacfwd(process_model, argnums=0) # df_dx
        self.H = jacfwd(observation_model, argnums=0) #dh_dx
        self.L = jacfwd(process_model, argnums=2) #df_dw
        self.M = jacfwd(observation_model, argnums=1) #dh_dv
        self.Q = process_noise_covariance
        self.R = observation_noise_covariance

    def __call__(self, x: jnp.ndarray, u: jnp.ndarray, z: jnp.ndarray, P: jnp.ndarray) -> Tuple[
        jnp.ndarray, jnp.ndarray]:
        # predict
        x_pred, P = self.predict(x, u, P)
        # update
        x, P = self.update(x_pred, P, z)
        return x, P

    def predict(self, x: jnp.ndarray, u: jnp.ndarray, P: jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray]:
        w = jnp.zeros_like(x)
        x_pred = self.f(x, u, w) # predict next state
        P = self.F(x, u, w) @ P @ self.F(x, u, w).transpose() + self.L(x, u, w) @ self.Q @ self.L(x, u, w).transpose()  # compute state covariance
        return x_pred, P

    def update(self, x: jnp.ndarray, P: jnp.ndarray, z: jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray]:
        v = jnp.zeros_like(z)
        y = z - self.h(x, v)  # innovation
        S = self.H(x, v) @ P @ self.H(x, v).transpose() + self.M(x, v) @ self.R @ self.M(x, v).transpose()  # innovation covariance
        K = P @ self.H(x, v).transpose() @ jnp.linalg.inv(S)  # kalman gain
        x_estimated = x + K @ y  # correct prediction
        P_estimated = (jnp.eye(x.shape[0]) - K @ self.H(x, v)) @ P  # correct covariance
        return x_estimated, P_estimated

class KalmanFilter(ExtendedKalmanFilter):

    def __init__(self,
                 A: jnp.ndarray,
                 B: jnp.ndarray,
                 H: jnp.ndarray,
                 process_noise_covariance: jnp.ndarray,
                 observation_noise_covariance: jnp.ndarray
                 ):
        process_model = lambda x, u, w: A @ x + B @ u + w
        observation_model = lambda x, v: H @ x + v
        super().__init__(process_model, observation_model, process_noise_covariance, observation_noise_covariance)

