"""Terminal entry point for the constrained dynamic optimization phase."""
import argparse
from pathlib import Path

from scripts.constrained_dynamic_model import OUT, main

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the constrained dynamic simulation.")
    parser.add_argument("--output-dir", type=Path, default=OUT,
                        help="Directory for simulation evidence (defaults to the repository evidence folder).")
    main(parser.parse_args().output_dir)
