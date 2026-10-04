import numpy as np

from count_tracker_ultratight import ultratight_mask


def test_ultratight_boundaries_are_applied_as_published():
    masks = ultratight_mask(
        pt=[0.6, 0.5, 0.6, 0.6, 0.6, 0.6],
        n_hits=[9, 9, 8, 9, 9, 9],
        n_holes=[2, 2, 2, 3, 2, 2],
        n_outliers=[3, 3, 3, 3, 4, 3],
        reduced_chi2=[2.9, 2.9, 2.9, 2.9, 2.9, 3.0],
    )

    assert np.array_equal(
        masks["ultratight"],
        np.array([True, False, False, False, False, False]),
    )
    assert np.array_equal(
        masks["all_except_outliers"],
        np.array([True, False, False, False, True, False]),
    )
