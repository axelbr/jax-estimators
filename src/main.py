from functools import partial
from estimation import ExtendedKalmanFilter, KalmanFilter

import jax.numpy as jnp
import numpy as np
from jax import jacfwd, jit
import matplotlib.pyplot as plt

from src.environments import DiffDriveRobot
from util import plot, History

def _motion_model(x, u, w, dt=0.01):
    return jnp.array([
        x[0] + x[3] * jnp.cos(x[2]) * dt,
        x[1] + x[3] * jnp.sin(x[2]) * dt,
        x[2] + u[1] * dt,
        u[0]
    ]) + w

def _observation_model(x, v):
    H = jnp.array([[1, 0, 0, 0],
                   [0, 1, 0, 0]])
    return H @ x + v

def controller(x):
    v = 1.0  # [m/s]
    yawrate = 0.1  # [rad/s]
    u = jnp.array([v, yawrate])
    return u

def main():
    env = DiffDriveRobot()
    z = env.reset()
    x_hat = jnp.zeros(4) # [x, y, yaw, velocity]
    x_cov = jnp.eye(4)

    Q = jnp.diag(jnp.array([
        0.1,  # variance of location on x-axis
        0.1,  # variance of location on y-axis
        jnp.deg2rad(0.5),  # variance of yaw angle
        0.1  # variance of velocity
    ])) ** 2  # predict state covariance
    R = jnp.diag(jnp.array([2, 2])) ** 2  # Observation x,y position covariance

    filter = ExtendedKalmanFilter(
        process_model=_motion_model,
        observation_model=_observation_model,
        process_noise_covariance=Q,
        observation_noise_covariance=R
    )
    filter = jit(filter)

    history = History()
    history.update(x=x_hat, z=z, x_hat=x_hat, covariance=x_cov)

    for t in range(5000):
        print(t)
        u = controller(x_hat) # [velocity, yaw_rate]
        obs, _, _, info = env.step(u)
        x_hat, x_cov = filter(x=x_hat, P=x_cov, u=u, z=obs)
        history.update(x=info['x'], z=obs, x_hat=x_hat, covariance=x_cov)

        if t % 100 == 0:
            plot(data=history)



if __name__ == '__main__':
    main()