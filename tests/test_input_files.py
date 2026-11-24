"""Preserve the board and associate proposals with one input revision."""

from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest.mock import patch

from kicad_pcb_physics import cli


class InputFiles(unittest.TestCase):
    def test_changed_inputs_cancel_proposal(self):
        for target in ("board", "settings"):
            with self.subTest(target=target), tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                board = root / "coupon.kicad_pcb"
                settings = root / "conditions.json"
                board.write_bytes(b"initial board")
                settings.write_text("{}")
                proposal = root / "proposal.kicad_pcb"
                output = root / "output"
                output.mkdir()
                (output / "results.json").write_text("previous results")
                (output / "provenance.json").write_text("previous provenance")

                def modify(snapshot, configuration, config_path, staged, digest):
                    self.assertEqual(snapshot.read_bytes(), b"initial board")
                    (staged / "results.json").write_text("new results")
                    path = board if target == "board" else settings
                    path.write_text("updated by another program")
                    self.assertEqual(snapshot.read_bytes(), b"initial board")
                    return {"stitching": []}, "test"

                with patch.object(cli, "analyze_snapshot", side_effect=modify):
                    with self.assertRaisesRegex(ValueError, "Inputs changed"):
                        cli.run(
                            board,
                            settings,
                            root / "output",
                            proposal=proposal,
                            stitch_case="trial",
                        )
                self.assertFalse(proposal.exists())
                self.assertEqual(
                    (output / "results.json").read_text(), "previous results"
                )
                self.assertEqual(
                    (output / "provenance.json").read_text(), "previous provenance"
                )

    def test_geometry_export_rejects_board_and_hardlink(self):
        try:
            kip = cli.interpreter({})
        except FileNotFoundError:
            self.skipTest("KiCad Python unavailable")
        with tempfile.TemporaryDirectory() as tmp:
            board = Path(tmp) / "sample.kicad_pcb"
            board.write_bytes(b"preserve this input")
            alias = Path(tmp) / "alias.json"
            alias.hardlink_to(board)
            for output in (board, alias):
                p = subprocess.run(
                    [
                        kip,
                        "-B",
                        Path(cli.__file__).with_name("bridge.py"),
                        board,
                        output,
                    ],
                    capture_output=True,
                    text=True,
                )
                self.assertNotEqual(p.returncode, 0)
                self.assertIn("separate JSON", p.stderr)
                self.assertEqual(board.read_bytes(), b"preserve this input")
