"""Exercise the installed CLI with an independently generated board and a real simulator."""

import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest
import numpy as np
from kicad_pcb_physics.cli import interpreter
from kicad_pcb_physics import spice


class EndToEnd(unittest.TestCase):
    def test_generated_board(self):
        try:
            kip = interpreter({})
        except FileNotFoundError:
            self.skipTest("KiCad Python unavailable")
        if not (os.environ.get("NGSPICE_BIN") or shutil.which("ngspice")):
            self.skipTest("ngspice unavailable")
        root = Path(__file__).resolve().parents[1]
        with tempfile.TemporaryDirectory() as work:
            w = Path(work)
            subprocess.run([kip, "-B", root / "examples/generate.py", w], check=True)
            subprocess.run(
                [
                    sys.executable,
                    "-B",
                    "-m",
                    "kicad_pcb_physics.cli",
                    w / "fixture.kicad_pcb",
                    w / "settings.json",
                    w / "output",
                ],
                check=True,
            )
            r = json.loads((w / "output/results.json").read_text())
            self.assertAlmostEqual(
                r["transients"][0]["measurements"]["received"],
                1 - np.exp(-1),
                delta=2e-5,
            )
            self.assertEqual(len(r["pdn"]), 2)
            for p in r["pdn"]:
                self.assertLess(p["spice"]["max_relative_error"], 1e-5)
            self.assertNotEqual(r["pdn"][0]["ripple_pp_V"], r["pdn"][1]["ripple_pp_V"])
            self.assertGreater(r["electrical"][0]["near_field"]["max_B_T"], 0.0)
            self.assertGreater(len(r["electrical"][1]["rows"][-1]["via_current"]), 0)
            self.assertTrue(any(g["missing_mm"] > 1 for g in r["returns"][0]["gaps"]))
            self.assertGreater(len(r["exposure"][0]["terminal_emf"]), 0)
            self.assertGreater(r["thermal"][0]["peak_board_c"], 25)
            self.assertLess(r["thermal"][0]["balance_error_w"], 1e-7)
            n = r["receivers"][0]["rows"][0]
            self.assertGreater(n["terminal_V"], 0)
            self.assertTrue((w / "output/index.html").is_file())
            if os.environ.get("LTSPICE_BIN"):
                cfg = json.loads((w / "settings.json").read_text())
                for p, spec in zip(r["pdn"], cfg["pdn"]):
                    verify = spice.check(p, spec["feed"], {"backend": "ltspice"})
                    self.assertLess(verify["max_relative_error"], 1e-5)
