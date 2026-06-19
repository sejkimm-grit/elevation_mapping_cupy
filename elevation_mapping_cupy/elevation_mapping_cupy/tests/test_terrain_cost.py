import numpy as np
import pytest

cp = pytest.importorskip("cupy")

from elevation_mapping_cupy.terrain_cost import traversability_to_terrain_cost


def test_traversability_to_terrain_cost_inverts_and_handles_unknown():
    traversability = cp.asarray([[1.0, 0.25, 0.0, cp.nan]], dtype=cp.float32)

    actual = cp.asnumpy(
        traversability_to_terrain_cost(
            traversability,
            unknown_cost=75.0,
            scale=100.0,
            offset=0.0,
            xp=cp,
        )
    )

    expected = np.asarray([[0.0, 75.0, 100.0, 75.0]], dtype=np.float32)
    np.testing.assert_allclose(actual, expected)
