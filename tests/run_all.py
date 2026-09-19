"""
Run every test suite and summarise.

    python tests/run_all.py
    python tests/run_all.py --fast     # skip the slow simulation-heavy suites

Each suite is a standalone script rather than a pytest module, so each one runs
in its own process. That is deliberate: the suites stub out cv2 and ultralytics
at import time, and sharing an interpreter between them would let one suite's
stubs leak into another's - which is exactly the kind of cross-contamination
that makes a passing test meaningless.
"""

import argparse
import subprocess
import sys
import time
from pathlib import Path

TESTS = Path(__file__).resolve().parent
ROOT = TESTS.parent

SUITES = [
    ("test_phase1.py", "detection, tracking, line counting, config", False),
    ("test_phase2.py", "schema, UTC handling, bucketing, queries", False),
    ("test_phase3.py", "signal safety invariants, simulator", True),
    ("test_phase3b.py", "features, leakage, baselines, ML strategy", True),
    ("test_phase4.py", "API contract, override safety", False),
    ("test_phase5.py", "PCU, Webster timing, three-way demo", True),
]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fast", action="store_true",
                        help="skip suites that run long simulations")
    args = parser.parse_args()

    print("=" * 66)
    total_passed = total_failed = 0
    failed_suites = []

    for filename, description, slow in SUITES:
        if args.fast and slow:
            print(f"  SKIP  {filename:<18} {description} (slow)")
            continue

        began = time.time()
        proc = subprocess.run(
            [sys.executable, str(TESTS / filename)],
            capture_output=True, text=True, cwd=str(ROOT),
        )
        elapsed = time.time() - began

        summary = next((ln for ln in reversed(proc.stdout.splitlines())
                        if "passed," in ln), "").strip()
        passed = failed = 0
        if summary:
            try:
                parts = summary.replace(",", "").split()
                passed = int(parts[parts.index("passed") - 1])
                failed = int(parts[parts.index("failed") - 1])
            except (ValueError, IndexError):
                pass

        total_passed += passed
        total_failed += failed
        mark = "ok  " if proc.returncode == 0 else "FAIL"
        print(f"  {mark}  {filename:<18} {passed:>3} passed  {failed:>2} failed  "
              f"{elapsed:>5.1f}s   {description}")

        if proc.returncode != 0:
            failed_suites.append(filename)
            for line in proc.stdout.splitlines():
                if line.strip().startswith("FAIL"):
                    print(f"          {line.strip()}")
            if proc.stderr.strip():
                print(f"          stderr: {proc.stderr.strip().splitlines()[-1][:120]}")

    print("=" * 66)
    print(f"  {total_passed} passed, {total_failed} failed"
          + (f"  ({len(failed_suites)} suite(s) failing: {', '.join(failed_suites)})"
             if failed_suites else ""))
    print("=" * 66)
    sys.exit(1 if failed_suites else 0)


if __name__ == "__main__":
    main()
