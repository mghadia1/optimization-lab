"""The suite is only worth its count if it fails when the code is wrong.

`tools/mutation_check.py` injects deliberate arithmetic faults one at a time and
records whether the suite notices. This test makes that a gate rather than a
script someone remembers to run: if a future change makes the tests looser, a
fault goes uncaught here and CI says so.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import unittest

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
HARNESS = PROJECT_ROOT / "tools" / "mutation_check.py"


class MutationCoverageTests(unittest.TestCase):
    @unittest.skipIf(
        os.environ.get("OPTIMIZATION_LAB_MUTATION_CHILD") == "1",
        "running inside the harness's own child suite; nesting would not terminate",
    )
    def test_every_injected_fault_is_caught(self) -> None:
        completed = subprocess.run(
            [sys.executable, str(HARNESS), "--json"],
            cwd=PROJECT_ROOT,
            capture_output=True,
            text=True,
        )
        self.assertTrue(completed.stdout.strip(), f"harness produced no report: {completed.stderr}")
        report = json.loads(completed.stdout)

        # A control run that already fails would make every mutation look caught.
        # The harness exits non-zero in that case, so this only needs the flag.
        self.assertTrue(report["control_passed"], f"control run failed: {report['control']}")

        missed = [row["id"] for row in report["results"] if not row["caught"]]
        self.assertEqual(
            missed,
            [],
            f"{len(missed)} injected fault(s) went unnoticed by the suite: {missed}",
        )
        self.assertEqual(report["caught"], report["mutations"])
        self.assertGreaterEqual(report["mutations"], 10)


if __name__ == "__main__":
    unittest.main()
