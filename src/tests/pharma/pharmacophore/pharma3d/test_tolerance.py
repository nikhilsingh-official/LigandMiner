import unittest

from rdkit.Chem.Pharm3D.Pharmacophore import Pharmacophore

from src.pharma.pharmacophore.pharma3d.tolerance import create_tolerance

from src.tests.pharma.pharmacophore.pharma3d._helpers import feature


class TestTolerance(unittest.TestCase):
    def test_expands_every_pairwise_interval_in_place(self):
        pharmacophore = Pharmacophore([feature(0), feature(2), feature(5)])
        pharmacophore.setLowerBound(0, 1, 1.5)
        pharmacophore.setUpperBound(0, 1, 2.5)
        pharmacophore.setLowerBound(0, 2, 4.0)
        pharmacophore.setUpperBound(0, 2, 6.0)

        returned = create_tolerance(pharmacophore, tolerance=0.5)

        self.assertIs(returned, pharmacophore)
        self.assertEqual(pharmacophore.getLowerBound(0, 1), 1.0)
        self.assertEqual(pharmacophore.getUpperBound(0, 1), 3.0)
        self.assertEqual(pharmacophore.getLowerBound(0, 2), 3.5)
        self.assertEqual(pharmacophore.getUpperBound(0, 2), 6.5)

    def test_lower_bounds_are_clamped_at_zero(self):
        pharmacophore = Pharmacophore([feature(0), feature(1)])
        pharmacophore.setLowerBound(0, 1, 0.25)
        pharmacophore.setUpperBound(0, 1, 1.0)

        create_tolerance(pharmacophore, tolerance=0.5)

        self.assertEqual(pharmacophore.getLowerBound(0, 1), 0.0)
        self.assertEqual(pharmacophore.getUpperBound(0, 1), 1.5)

    def test_single_feature_is_unchanged(self):
        pharmacophore = Pharmacophore([feature(0)])
        self.assertIs(create_tolerance(pharmacophore), pharmacophore)


if __name__ == "__main__":
    unittest.main()
