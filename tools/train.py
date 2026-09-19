"""
Train and evaluate traffic-flow forecasters against honest baselines.

    python tools/train.py --camera north --horizon 15
    python tools/train.py --camera north --model gbr        # no TensorFlow needed
    python tools/train.py --camera north --model both

WHAT THIS PRINTS AND HOW TO READ IT:

Every model is scored on the same held-out test split, in vehicles per minute,
alongside three baselines. The column that decides whether the work was
worthwhile is SKILL - the percentage by which a model beats the seasonal-naive
baseline ("same minute yesterday").

    skill > 0    the model knows something the calendar does not
    skill ~ 0    it reproduced the daily pattern and added nothing
    skill < 0    it is worse than doing nothing clever, and should not ship

A negative skill score is a legitimate result, not a failure of the project. It
says this junction's traffic is dominated by its daily rhythm, which is worth
knowing and worth writing down. What is not legitimate is reporting an MAE with
no baseline beside it.
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.database.db_handler import Database, TrafficRepository
from src.models.baselines import (evaluate, poisson_noise_floor, run_baselines,
                                  skill_score)
from src.models.features import build_dataset
from src.utils.config_loader import load_config
from src.utils.logger import setup_logging


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--camera", default="north", help="approach to model")
    parser.add_argument("--horizon", type=int, default=15, help="minutes ahead to predict")
    parser.add_argument("--lookback", type=int, default=30, help="minutes of history per sample")
    parser.add_argument("--hours", type=int, default=24 * 30, help="history to load")
    parser.add_argument("--model", choices=("lstm", "gbr", "both"), default="both")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--save", default=None, help="path to save the trained LSTM")
    parser.add_argument("--config", default="config/config.yaml")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    setup_logging(level="INFO" if args.verbose else "WARNING")

    config = load_config(args.config, env_file=None)
    repo = TrafficRepository(Database(config.database_url()))
    rows = repo.per_minute(args.camera, hours=args.hours)

    if not rows:
        raise SystemExit(
            f"No history for camera '{args.camera}'. Generate some first:\n"
            f"  python tools/simulate.py generate --days 21"
        )

    dataset = build_dataset(rows, lookback=args.lookback, horizon=args.horizon)
    print(f"\nCamera '{args.camera}' | {len(rows)} minutes of history")
    print(f"Predicting flow {args.horizon} minutes ahead from {args.lookback} minutes of history")
    print(dataset.describe())

    results = run_baselines(dataset)
    reference = results["seasonal_naive"]

    if args.model in ("gbr", "both"):
        from src.models.lstm_model import GradientBoostingForecaster
        gbr = GradientBoostingForecaster().fit(dataset)
        results["gbr"] = evaluate(dataset.y_test_raw, gbr.predict(dataset))

    if args.model in ("lstm", "both"):
        from src.models.lstm_model import LSTMForecaster
        lstm = LSTMForecaster(epochs=args.epochs)
        lstm.fit(dataset, verbose=1 if args.verbose else 0)
        results["lstm"] = evaluate(dataset.y_test_raw, lstm.predict(dataset))
        if args.save:
            lstm.save(args.save, dataset)

    print(f"\n{'model':<16}{'MAE':>8}{'RMSE':>8}{'bias':>8}{'R2':>8}{'skill':>9}")
    print("-" * 57)
    order = ["mean", "persistence", "seasonal_naive", "gbr", "lstm"]
    for name in order:
        if name not in results:
            continue
        m = results[name]
        skill = skill_score(m, reference)
        marker = "  <- baseline" if name == "seasonal_naive" else ""
        print(f"{name:<16}{m.mae:>8.3f}{m.rmse:>8.3f}{m.bias:>+8.3f}"
              f"{m.r2:>8.3f}{skill:>+8.1f}%{marker}")
    print("-" * 57)
    print("MAE and RMSE are vehicles per minute. Skill is improvement over "
          "seasonal-naive.\n")

    floor = poisson_noise_floor(dataset.y_test_raw)
    best_mae = min(m.mae for m in results.values())
    print(f"Irreducible noise floor: MAE {floor:.3f} (a perfect oracle knowing the true "
          f"arrival rate)")
    print(f"Best model is {(best_mae - floor) / floor * 100:+.1f}% above that floor - "
          f"{'little headroom remains' if best_mae < floor * 1.25 else 'there is signal left to extract'}.\n")

    learned = {n: skill_score(results[n], reference) for n in ("gbr", "lstm") if n in results}
    if learned:
        best, best_skill = max(learned.items(), key=lambda kv: kv[1])
        if best_skill > 5:
            print(f"VERDICT: {best} beats 'same minute yesterday' by {best_skill:.1f}%. "
                  f"The learned model is earning its complexity.")
        elif best_skill > -5:
            print(f"VERDICT: {best} is within {abs(best_skill):.1f}% of the naive baseline. "
                  f"This traffic is dominated by its daily cycle; the model is "
                  f"reproducing the calendar rather than adding to it. Report that "
                  f"honestly - it is a finding, not a failure.")
        else:
            print(f"VERDICT: every learned model is WORSE than seasonal-naive "
                  f"(best {best_skill:.1f}%). Do not deploy it. Likely causes: too "
                  f"little history, a horizon longer than the signal in the data, "
                  f"or overfitting - check the train/val gap with --verbose.")
        print()


if __name__ == "__main__":
    main()
