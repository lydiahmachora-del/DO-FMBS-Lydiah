"""Terminal entry point for the constrained dynamic optimization phase."""
import argparse
from pathlib import Path

from scripts.constrained_dynamic_model import COST_WEIGHTS, OUT, PREFERRED_Y, TARGET_Y, main

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the constrained dynamic simulation.")
    parser.add_argument("--output-dir", type=Path, default=OUT,
                        help="Directory for simulation evidence (defaults to the repository evidence folder).")
    parser.add_argument("--rounds", type=int, default=12)
    parser.add_argument("--candidates-per-round", type=int, default=63)
    parser.add_argument("--target-y", type=float, default=TARGET_Y)
    parser.add_argument("--preferred-target-y", type=float, default=PREFERRED_Y)
    parser.add_argument("--defense-cost-weight", type=float, default=COST_WEIGHTS["defense"])
    parser.add_argument("--participation-cost-weight", type=float, default=COST_WEIGHTS["participation"])
    parser.add_argument("--defense-switch-cost-weight", type=float, default=COST_WEIGHTS["defense_switch"])
    parser.add_argument("--participation-switch-cost-weight", type=float, default=COST_WEIGHTS["participation_switch"])
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    main(args.output_dir, rounds=args.rounds,
         candidates_per_round=args.candidates_per_round,
         target=args.target_y, preferred=args.preferred_target_y, seed=args.seed,
         cost_weights={"defense": args.defense_cost_weight,
                       "participation": args.participation_cost_weight,
                       "defense_switch": args.defense_switch_cost_weight,
                       "participation_switch": args.participation_switch_cost_weight})
