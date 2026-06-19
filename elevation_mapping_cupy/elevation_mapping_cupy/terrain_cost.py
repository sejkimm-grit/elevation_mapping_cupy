import numpy as np


def traversability_to_terrain_cost(
    traversability,
    unknown_cost=75.0,
    scale=100.0,
    offset=0.0,
    xp=np,
):
    traversability = xp.asarray(traversability, dtype=xp.float32)
    cost = (1.0 - traversability) * scale + offset
    cost = xp.clip(cost, 0.0, 100.0)
    cost = xp.where(xp.isfinite(traversability), cost, unknown_cost)
    return cost.astype(xp.float32)
