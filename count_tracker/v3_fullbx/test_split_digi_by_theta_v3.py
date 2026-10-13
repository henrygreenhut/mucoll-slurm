"""Unit checks for the native v3 theta selection."""

import unittest

from split_digi_by_theta_v3 import in_region, theta_degrees


class Position:
    def __init__(self, x, y, z):
        self.x, self.y, self.z = x, y, z


class Hit:
    def __init__(self, x, y, z):
        self.position = Position(x, y, z)

    def getPosition(self):
        return self.position


class ThetaSelectionTests(unittest.TestCase):
    def test_polar_angle_uses_detector_position(self):
        self.assertAlmostEqual(theta_degrees(Hit(0, 0, 1)), 0.0)
        self.assertAlmostEqual(theta_degrees(Hit(1, 0, 0)), 90.0)
        self.assertAlmostEqual(theta_degrees(Hit(0, 0, -1)), 180.0)

    def test_regions_are_disjoint_and_cover_boundaries(self):
        regions = ((0, 30), (30, 70), (70, 110), (110, 150), (150, 180))
        for theta, expected in ((0, 0), (29.999, 0), (30.001, 1),
                                (69.999, 1), (70.001, 2), (90, 2),
                                (109.999, 2), (110.001, 3),
                                (149.999, 3), (150.001, 4), (180, 4)):
            radians = theta * 3.141592653589793 / 180.0
            hit = Hit(__import__("math").sin(radians), 0,
                      __import__("math").cos(radians))
            selected = [index for index, bounds in enumerate(regions)
                        if in_region(hit, *bounds)]
            self.assertEqual(selected, [expected])


if __name__ == "__main__":
    unittest.main()
