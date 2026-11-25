import unittest

from kicad_pcb_physics.calibration import calibrate
from examples.generate_calibration import experiment


class CalibrationTests(unittest.TestCase):
    def test_recovers_synthetic_material(self):
        data = experiment()
        result = calibrate(data)
        indices = result["accepted_indices"]
        self.assertEqual(len(indices), 1)
        self.assertEqual(data["runs"][indices[0]]["parameters"],
                         {"resistance": 32, "capacity": 1.5})
        target = result["targets"]["pulse"]
        expected = data["runs"][indices[0]]["predictions"]["pulse"]
        self.assertAlmostEqual(target["residual_transfer_range"][0], expected - 0.4)
        self.assertAlmostEqual(target["residual_transfer_range"][1], expected + 0.4)

    def test_unseen_error_does_not_change_selection(self):
        data = experiment()
        before = calibrate(data)
        check = next(row for row in data["observations"] if row["role"] == "check")
        check["low"] += 10
        check["high"] += 10
        after = calibrate(data)
        self.assertEqual(before["accepted_indices"], after["accepted_indices"])
        for a, b in zip(before["targets"]["pulse"]["residual_transfer_range"],
                        after["targets"]["pulse"]["residual_transfer_range"]):
            self.assertAlmostEqual(b - a, 10)

    def test_rejects_joint_fit_and_preserves_ties(self):
        data = experiment()
        data["observations"] = [data["observations"][1]]
        data["targets"][0]["error_from"] = []
        self.assertEqual(len(calibrate(data)["accepted_indices"]), 4)
        data["observations"][0].update(low=100, high=101)
        result = calibrate(data)
        self.assertFalse(result["targets"])
        self.assertGreater(result["minimum_max_fit_miss"], 0)

    def test_incompatible_units_and_invalid_data(self):
        data = experiment()
        data["targets"][0]["unit"] = "m"
        with self.assertRaises(ValueError):
            calibrate(data)
        data = experiment()
        data["runs"][0]["predictions"]["warm"] = float("nan")
        with self.assertRaises(ValueError):
            calibrate(data)
