import functools
from typing import Callable

import jax
import jax.numpy as jnp


@functools.partial(jax.custom_jvp, nondiff_argnums=(1,))
def _sqrt(x, tol=0.0):
    return jnp.sqrt(jnp.maximum(x, tol))


@_sqrt.defjvp
def _sqrt_jvp(tol, primals, tangents):
    (x,) = primals
    (x_dot,) = tangents
    safe_tol = max(tol, 1e-30)
    square_root = _sqrt(x, safe_tol)
    return square_root, jnp.where(x > safe_tol, x_dot / (2 * square_root), 0.0)


def absolute_reward_diff(r1: jax.Array, r2: jax.Array) -> jax.Array:
    """Returns |r1 - r2|."""

    return jnp.abs(r1 - r2)


def cosine_distance(x: jax.Array, y: jax.Array) -> jax.Array:
    """Cosine distance."""

    numerator = jnp.sum(x * y)
    denominator = jnp.sqrt(jnp.sum(x**2)) * jnp.sqrt(jnp.sum(y**2))
    cos_similarity = numerator / (denominator + 1e-9)
    return jnp.arctan2(_sqrt(1.0 - cos_similarity**2), cos_similarity)


def make_squarelike(x: jax.Array) -> jax.Array:
    """Makes into square matrix."""

    B = x.shape[0]
    if x.ndim > 1:
        R = x.shape[-1]
        return jnp.reshape(jnp.tile(x, B), (B, B, R))

    return jnp.reshape(jnp.tile(x, B), (B, B))


def representation_distances(
    first_reprs: jax.Array,
    second_reprs: jax.Array,
    distance_fn: Callable[..., jax.Array],
    beta: float = 0.1,
) -> jax.Array:
    """Representation distances."""

    B = first_reprs.shape[0]
    R = first_reprs.shape[-1]

    first_squared_reprs = make_squarelike(first_reprs)
    first_squared_reprs = jnp.reshape(first_squared_reprs, [B**2, R])

    second_squared_reprs = make_squarelike(second_reprs)
    second_squared_reprs = jnp.transpose(second_squared_reprs, [1, 0, 2])
    second_squared_reprs = jnp.reshape(second_squared_reprs, [B**2, R])

    base_distances = jax.vmap(distance_fn, in_axes=(0, 0))(
        first_squared_reprs, second_squared_reprs
    )
    norm_average = 0.5 * (
        jnp.sum(jnp.square(first_squared_reprs), -1)
        + jnp.sum(jnp.square(second_squared_reprs), -1)
    )
    return norm_average + beta * base_distances


def target_distances(
    representations: jax.Array,
    rewards: jax.Array,
    distance_fn: Callable[..., jax.Array],
    gamma: float,
) -> jax.Array:
    """Target distances."""

    next_state_similarities = representation_distances(
        representations, representations, distance_fn=distance_fn
    )
    squared_rewards = make_squarelike(rewards)
    squared_rewards_transp = jnp.transpose(squared_rewards)

    squared_rewards = squared_rewards.reshape((squared_rewards.shape[0] ** 2))
    squared_rewards_transp = squared_rewards_transp.reshape(
        (squared_rewards_transp.shape[0] ** 2)
    )

    reward_diffs = absolute_reward_diff(squared_rewards, squared_rewards_transp)
    return jax.lax.stop_gradient(reward_diffs + gamma * next_state_similarities)
