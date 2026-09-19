"""
Run the traffic simulator: compare control strategies, or generate training data.

TWO JOBS:

  compare   Runs fixed-time and adaptive against IDENTICAL traffic and prints
            the delay comparison. This produces the headline number for your
            report, and the numbers are reproducible because the seed is fixed.

                python tools/simulate.py compare --hours 12

  generate  Simulates several days and writes the snapshots to the database, so
            the LSTM has history to train on before you have filmed anything.

                python tools/simulate.py generate --days 14

BE HONEST ABOUT THIS IN THE REPORT. Synthetic data proves the pipeline works and
lets the model be built; it does not prove the model works on real traffic,
because the simulator generates exactly the pattern the model then "discovers".
Say so, and re-run on real data when you have it. A reviewer who spots
undisclosed synthetic results will discount everything else in the paper; one
who reads that you knew the limitation will not.
"""

import argparse
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.simulator import ApproachSim, TrafficSimulator, compare_strategies
from src.traffic_controller import (AdaptiveStrategy, FixedTimeStrategy,
                                    IntersectionController, Phase)
from src.utils.logger import setup_logging


def default_phases(max_green: float = 50.0):
    return [
        Phase(id="ns", name="North-South", camera_ids=["north", "south"],
              min_green=10.0, max_green=max_green, yellow=3.0, all_red=2.0),
        Phase(id="ew", name="East-West", camera_ids=["east", "west"],
              min_green=10.0, max_green=max_green, yellow=3.0, all_red=2.0),
    ]


def default_approaches():
    """
    A main road crossing a side road - the asymmetry is the point.

    Fixed timing handles asymmetric demand worst, because one split cannot suit
    both roads: sized for the main road it wastes green on the side road, sized
    for the side road it strands the main road. That gap is exactly what an
    adaptive controller recovers, so a symmetric junction would understate the
    benefit and a wildly asymmetric one would overstate it.
    """
    return [
        ApproachSim("north", lanes=2, peak_vehicles_per_hour=1250),
        ApproachSim("south", lanes=2, peak_vehicles_per_hour=1150),
        ApproachSim("east", lanes=1, peak_vehicles_per_hour=420),
        ApproachSim("west", lanes=1, peak_vehicles_per_hour=360),
    ]


def cmd_compare(args) -> None:
    results = compare_strategies(
        default_approaches, default_phases(args.max_green),
        {
            "fixed-25": FixedTimeStrategy(25),
            "fixed-35": FixedTimeStrategy(35),
            f"fixed-{int(args.max_green)}": FixedTimeStrategy(args.max_green),
            "adaptive": AdaptiveStrategy(queue_threshold=args.queue_threshold),
        },
        hours=args.hours, seed=args.seed, start_hour=args.start_hour,
    )

    print(f"\n{args.hours:g} simulated hours, seed {args.seed}, identical arrivals for every run\n")
    print(f"{'strategy':<12}{'avg delay':>11}{'served':>10}{'peak queue':>12}")
    print("-" * 45)
    for label, r in results.items():
        print(f"{label:<12}{r.average_delay:>10.1f}s{r.total_served:>10.0f}{r.peak_queue:>12.1f}")

    adaptive = results["adaptive"]
    best_fixed = min((r.average_delay, k) for k, r in results.items() if k.startswith("fixed"))
    saved = (best_fixed[0] - adaptive.average_delay) / best_fixed[0] * 100
    print("-" * 45)
    print(f"\nAdaptive vs best fixed schedule ({best_fixed[1]}): {saved:+.1f}% average delay")
    print(f"Total delay saved: {(best_fixed[0]-adaptive.average_delay)*adaptive.total_served/3600:.1f} "
          f"vehicle-hours over {args.hours:g}h\n")

    print("Per approach (adaptive):")
    for cam, d in adaptive.per_approach.items():
        print(f"   {cam:<7} delay {d['average_delay']:5.1f}s   served {d['served']:6.0f}   "
              f"peak queue {d['peak_queue']:5.1f}")

    print("\nWhy greens ended (adaptive):")
    for reason, n in sorted(adaptive.reason_counts.items(), key=lambda kv: -kv[1]):
        print(f"   {reason:<18} {n}")

    print("\nGreen durations (adaptive):")
    for pid, durations in adaptive.green_durations.items():
        if durations:
            print(f"   {pid:<4} n={len(durations):<5} mean {sum(durations)/len(durations):5.1f}s   "
                  f"range {min(durations):.0f}-{max(durations):.0f}s")
    print()


def cmd_generate(args) -> None:
    from src.database.db_handler import Database, SnapshotWriter
    from src.database.models import Camera
    from src.utils.config_loader import load_config

    config = load_config(args.config, env_file=args.env)
    db = Database(config.database_url())
    db.create_all()

    # The simulated approaches must exist as cameras, or the foreign key rejects
    # every snapshot.
    with db.session() as s:
        for cam in ("north", "south", "east", "west"):
            if s.get(Camera, cam) is None:
                s.add(Camera(id=cam, name=f"Simulated {cam} approach", direction=cam))

    writer = SnapshotWriter(db, bucket_seconds=args.bucket_seconds, batch_size=500)
    start = datetime.now(timezone.utc) - timedelta(days=args.days)
    total = 0

    for day in range(args.days):
        controller = IntersectionController(
            "sim_signal", default_phases(), AdaptiveStrategy())
        sim = TrafficSimulator(default_approaches(), controller,
                               start_hour=0.0, seed=args.seed + day)
        result = sim.run(hours=24.0, snapshot_interval=args.bucket_seconds,
                         start_time=start + timedelta(days=day))
        for snap in result.snapshots:
            writer.add(snap)
        written = writer.close()
        total += written
        print(f"  day {day+1}/{args.days}: {written} rows  "
              f"(avg delay {result.average_delay:.1f}s, served {result.total_served:.0f})")
        writer = SnapshotWriter(db, bucket_seconds=args.bucket_seconds, batch_size=500)

    print(f"\n{total} rows written to {config.database_url()}")
    print(f"That is {args.days} days at {args.bucket_seconds}s resolution across 4 approaches.")
    print("\nInspect it with:")
    print("  python -c \"from src.database.db_handler import *; from src.utils.config_loader import *; "
          "r=TrafficRepository(Database(load_config().database_url())); "
          "print(r.per_minute('north', hours=24)[:5])\"")
    db.dispose()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)

    c = sub.add_parser("compare", help="compare fixed-time against adaptive control")
    c.add_argument("--hours", type=float, default=12.0)
    c.add_argument("--seed", type=int, default=42)
    c.add_argument("--start-hour", type=float, default=6.0)
    c.add_argument("--max-green", type=float, default=50.0)
    c.add_argument("--queue-threshold", type=int, default=2)
    c.set_defaults(func=cmd_compare)

    g = sub.add_parser("generate", help="write simulated history to the database")
    g.add_argument("--days", type=int, default=14)
    g.add_argument("--seed", type=int, default=42)
    g.add_argument("--bucket-seconds", type=int, default=5)
    g.add_argument("--config", default="config/config.yaml")
    g.add_argument("--env", default=".env")
    g.set_defaults(func=cmd_generate)

    args = parser.parse_args()
    setup_logging(level="WARNING")
    args.func(args)


if __name__ == "__main__":
    main()
