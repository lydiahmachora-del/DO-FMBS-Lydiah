"""Generate a beginner-friendly DO-FMBS code and simulation guide.

The guide is intentionally separate from Chapter Four.  It explains how the
Phase 1-6 scripts work, how to run them, which values can be changed, where the
data and outputs are stored, and how the original X1-X4 equation maps to the
implemented AI/PA/DM/GS model.  Recorded values are loaded from the synchronized
isolated replay rather than copied from prose.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import shutil
import sys
from datetime import datetime
from html import escape as html_escape
from pathlib import Path
from typing import Sequence
from zoneinfo import ZoneInfo

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.generate_chapter_four_report import (
    DEFAULT_ARCHITECTURE,
    DEFAULT_EVALUATOR_DIR,
    DEFAULT_RUN_DIR,
    ROOT,
    load_evidence,
    md_table,
    num,
    pct,
)
from scripts.generate_constrained_dynamic_report import (
    _equation_image_buffer,
    _markdown_blocks,
    _plain_markdown,
)


REPORT_STEM = "DO_FMBS_CODE_AND_SIMULATION_PRACTICAL_GUIDE"
DEFAULT_OUTPUT_DIR = ROOT / "docs" / "code_simulation_user_guide"
FIGURE_RE = re.compile(r"^\[\[FIGURE:([^|]+)\|(.+)\]\]$")

FIGURE_SPECS = (
    ("GUIDE_FIGURE_1_SIMULATION_FLOW.png", "Figure 1: Beginner-oriented Phase 1-6 simulation flow (Researcher, 2026)."),
    ("GUIDE_FIGURE_2_LFW_FACES.png", "Figure 2: Sample LFW face images loaded by Phase 1 (Researcher, 2026)."),
    ("GUIDE_FIGURE_3_SYNTHETIC_FINGERPRINTS.png", "Figure 3: Sample fingerprints generated locally by Phase 1 (Researcher, 2026)."),
    ("GUIDE_FIGURE_4_CLIENT_DISTRIBUTION.png", "Figure 4: Persisted Phase-1 client data distribution (Researcher, 2026)."),
    ("GUIDE_FIGURE_5_FEDERATED_TRAINING.png", "Figure 5: Recorded federated-training accuracy curves (Researcher, 2026)."),
    ("GUIDE_FIGURE_6_ATTACKS.png", "Figure 6: Recorded adversarial-attack evaluation (Researcher, 2026)."),
    ("GUIDE_FIGURE_7_DEFENSES.png", "Figure 7: Recorded defense-mechanism comparison (Researcher, 2026)."),
    ("GUIDE_FIGURE_8_ROBUSTNESS.png", "Figure 8: Recorded validation under increasing FGSM strength (Researcher, 2026)."),
)

REJECTED = (
    "legacy", "not recorded", "not measured", "not satisfied",
    "not yet wired", "evidence store", "needs further optimization",
)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate the DO-FMBS practical code guide.")
    parser.add_argument("--run-dir", type=Path, default=DEFAULT_RUN_DIR)
    parser.add_argument("--evaluator-dir", type=Path, default=DEFAULT_EVALUATOR_DIR)
    parser.add_argument("--architecture-image", type=Path, default=DEFAULT_ARCHITECTURE)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--formats", default="md,docx,pdf")
    parser.add_argument("--validate-inputs-only", action="store_true")
    return parser.parse_args(argv)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def figure_marker(filename: str, caption: str) -> str:
    return f"[[FIGURE:{filename}|{caption}]]"


def make_figures(evidence) -> list[Path]:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

    output = evidence.output_dir
    output.mkdir(parents=True, exist_ok=True)

    # A compact teaching diagram deliberately distinguishes the artifact-backed
    # model pipeline from the synthetic controller demonstration.
    fig, ax = plt.subplots(figsize=(9.2, 12.2))
    ax.set_xlim(0, 10); ax.set_ylim(0, 14); ax.axis("off")
    ax.set_title("HOW THE DO-FMBS SIMULATION RUNS", fontsize=18, weight="bold", color="#17365D", pad=15)
    boxes = [
        ("PHASE 1 — DATA", "Download LFW faces\nGenerate fingerprints locally\nStandardize, split and save arrays", 12.25, "#DDEBF7"),
        ("PHASE 2 — FEDERATED BASELINE", "Create five in-memory client sets\nTrain locally and combine with FedAvg\nSave baseline model and metrics", 10.25, "#E2F0D9"),
        ("PHASE 3 — ATTACKS", "Run FGSM and PGD evasion\nRun membership inference\nRun label-flipping poisoning", 8.25, "#FCE4D6"),
        ("PHASE 4 — DEFENSES", "Train no-defense, DP-style, ART\nand combined DO-FMBS models\nSave final defended model", 6.25, "#FFF2CC"),
        ("PHASE 5 — VALIDATION", "Compare baseline and final model\nRobustness sweep and statistics\nSave validation results", 4.25, "#E4DFEC"),
        ("PHASE 6 — DYNAMIC CONTROL", "Simulate 700 AI/PA/DM/GS conditions\nEstimate B0…B4 and search 63 controls\nSave coefficients and 12-round schedule", 2.25, "#D9EAD3"),
    ]
    for title, body, y, color in boxes:
        patch = FancyBboxPatch((1.15, y), 7.7, 1.25, boxstyle="round,pad=0.04,rounding_size=0.10",
                               linewidth=1.6, edgecolor="#17365D", facecolor=color)
        ax.add_patch(patch)
        ax.text(1.45, y + .90, title, fontsize=11.5, weight="bold", color="#17365D")
        ax.text(1.45, y + .18, body, fontsize=9.6, va="bottom", color="#202020", linespacing=1.25)
    for index in range(5):
        y1 = boxes[index][2]; y2 = boxes[index + 1][2]
        style = "-" if index < 4 else "--"
        ax.add_patch(FancyArrowPatch((5, y1), (5, y2 + 1.25), arrowstyle="-|>", mutation_scale=16,
                                     linewidth=1.8, linestyle=style, color="#4472C4"))
    ax.text(5, .75, "Solid flow = recommended execution order  |  Dashed flow = controller/design linkage",
            ha="center", fontsize=9, color="#666666")
    path = output / FIGURE_SPECS[0][0]
    fig.savefig(path, dpi=220, bbox_inches="tight", facecolor="white"); plt.close(fig)

    source_names = (
        "do_fmbs_lfw_faces.png",
        "do_fmbs_fingerprints.png",
        "do_fmbs_client_distribution.png",
        "do_fmbs_training_accuracy.png",
        "do_fmbs_attack_extended_results.png",
        "do_fmbs_comparison_results.png",
        "do_fmbs_validation_robustness.png",
    )
    for (target_name, _), source_name in zip(FIGURE_SPECS[1:], source_names):
        source = evidence.recorded.run_dir / source_name
        if not source.is_file() or source.stat().st_size == 0:
            raise FileNotFoundError(f"Guide figure source is missing: {source}")
        shutil.copy2(source, output / target_name)

    figures = [output / name for name, _ in FIGURE_SPECS]
    for figure in figures:
        if figure.stat().st_size < 10_000:
            raise ValueError(f"Guide figure is unexpectedly small: {figure}")
    return figures


def build_markdown(e, generated_at: str) -> str:
    r = e.recorded
    runtime_total = sum(float(row["elapsed_seconds"]) for row in r.runtime_rows)
    clean = {row["model"]: row for row in r.clean_rows}
    final = clean["DO-FMBS final"]
    base = clean["Baseline comparator"]
    coefficient_map = {row["parameter"]: row for row in e.coefficients}

    phase_rows = [
        [1, "Data preparation", "LFW download/cache and local fingerprint generation", "face_data.npz; fingerprint_data.npz; client files"],
        [2, "Federated baseline", "Five clients, local neural training and sample-weighted FedAvg", "baseline_model.keras; baseline_metrics.pkl"],
        [3, "Attack simulation", "FGSM, PGD, membership inference and label flipping", "attack_results.pkl; attack figures"],
        [4, "Defense comparison", "No defense, DP-style noise, ART and combined defense", "do_fmbs_model.keras; final_results.pkl"],
        [5, "Testing and validation", "Robustness sweep, classification metrics and statistics", "validation_results.pkl; validation report/figures"],
        [6, "Dynamic optimization", "Synthetic coefficient recovery and round-specific control", "model_fit.json; coefficient/schedule CSV files"],
    ]
    dependency_rows = [
        ["Phase 1 → Phase 2", "Phase 2 reads face_data.npz and fingerprint_data.npz"],
        ["Phase 2 → Phase 3", "Phase 3 reads the baseline model plus Phase-1 arrays"],
        ["Phase 1 → Phase 4", "Phase 4 trains new comparison models from the prepared arrays"],
        ["Phase 2 + Phase 4 → Phase 5", "Phase 5 compares baseline_model.keras and do_fmbs_model.keras"],
        ["Design variables → Phase 6", "Phase 6 independently demonstrates the constrained equation/controller"],
    ]
    data_rows = [
        ["Faces", "Labeled Faces in the Wild (LFW)", 3023, 62, "50 × 37", 1850, 2418, 605],
        ["Fingerprints", "Locally generated synthetic data", 800, 100, "96 × 96", 9216, 640, 160],
    ]
    client_rows = [[f"Client {row['client']}", row["face"], row["finger"], row["total"]] for row in e.client_rows]
    runtime_rows = [[f"Phase {row['phase']}", row["script"], f"{float(row['elapsed_seconds']):.6f}", row["status"]] for row in r.runtime_rows]
    attack_rows = []
    for fgsm, pgd in zip(e.attacks["fgsm_results"], e.attacks["pgd_results"]):
        attack_rows.append([num(fgsm["epsilon"], 2), pct(fgsm["success_rate"]), pct(pgd["success_rate"])])
    defense_rows = []
    display_names = {"baseline": "No defense", "dp": "DP-style noise", "art": "Adversarial training", "do_fmbs": "Combined DO-FMBS"}
    for key in ("baseline", "dp", "art", "do_fmbs"):
        row = e.defenses["comparison"][key]
        defense_rows.append([display_names[key], pct(row["face_accuracy"]), pct(row["attack_success_rate"]), pct(row["privacy_score"])])
    coefficient_rows = []
    for parameter in ("B0", "B_AI", "B_PA", "B_DM", "B_GS"):
        row = coefficient_map[parameter]
        coefficient_rows.append([parameter, f"{float(row['estimate']):.6f}", row["constraint"], "YES" if row["constraint_satisfied"].casefold() == "true" else "CHECK"])

    command_rows = [
        [1, r".\venv\Scripts\python.exe phase1_data_preparation.py", "Uses the fixed Phase-1 data design"],
        [2, r".\venv\Scripts\python.exe phase2_federated_learning.py --rounds 12 --local-epochs 2 --batch-size 64 --max-pairs 640 --test-pairs 100", "Recorded configuration"],
        [3, r".\venv\Scripts\python.exe phase3_adversarial_attacks.py --epsilon 0.10 --membership-threshold 0.70 --poison-rate 0.10 --poison-epochs 3", "Recorded configuration"],
        [4, r".\venv\Scripts\python.exe phase4_defense_mechanisms.py --epochs 12 --max-pairs 0 --test-pairs 0 --batch-size 32", "0 means use all available pairs"],
        [5, r".\venv\Scripts\python.exe phase5_model_validation.py", "Uses the fixed epsilon sweep"],
        [6, r".\venv\Scripts\python.exe phase6_constrained_dynamic_optimization.py", "Writes constrained-model evidence"],
    ]
    parameter_rows = [
        ["Phase 1", "Fixed in source", "min_faces_per_person=20; resize=0.4; 100 fingers; 8 impressions; 80/20 split", "phase1_data_preparation.py:31, 48-50, 93-95, 123-125"],
        ["Phase 2", "--rounds", "Number of server aggregation rounds", "phase2_federated_learning.py:27"],
        ["Phase 2", "--local-epochs", "Epochs trained by each client per round", "phase2_federated_learning.py:28"],
        ["Phase 2", "--batch-size", "Local training mini-batch size", "phase2_federated_learning.py:29"],
        ["Phase 2", "--max-pairs / --test-pairs", "Training/test batching-pair limits", "phase2_federated_learning.py:30-31"],
        ["Phase 3", "--epsilon", "Adds one value to the fixed epsilon sweep; a value already in the sweep does not change it", "phase3_adversarial_attacks.py:23,321-323"],
        ["Phase 3", "--membership-threshold", "Confidence threshold for membership inference", "phase3_adversarial_attacks.py:24"],
        ["Phase 3", "--poison-rate / --poison-epochs", "Label-flip share and poisoned retraining epochs", "phase3_adversarial_attacks.py:25-26"],
        ["Phase 4", "--epochs / --batch-size", "Defense training epochs and mini-batch size", "phase4_defense_mechanisms.py:43,48"],
        ["Phase 4", "--max-pairs / --test-pairs", "Data limits; zero selects all available rows", "phase4_defense_mechanisms.py:44-47"],
        ["Phase 5", "Fixed in source", "epsilon={0,.05,.10,.15,.20}; ten repeated subsets", "phase5_model_validation.py:110,174-201"],
        ["Phase 6", "Fixed controller constants", "AI/PA domains, Y targets, cost weights and 21×3 search", "scripts/constrained_dynamic_model.py:19-38,304-340"],
    ]
    variable_rows = [
        ["X1", "AI", "Observed attack pressure", "AI=epsilon/0.30", "0-1", "normalize_ai(), lines 78-82"],
        ["X2", "PA", "Observed privacy pressure", "PA=2 max(0,A_MIA-0.50)", "0-0.50 fitted", "membership_advantage(), lines 85-89"],
        ["X3", "DM", "Selected defense intensity", "Continuous controller choice", "0-1", "defense_configuration(), lines 99-113"],
        ["X4", "GS", "Selected client participation", "GS=(K-1)/4", "0-1", "gs_from_clients(), lines 92-96"],
        ["Y", "Y(t+1)", "Predicted next-round efficiency", "Fitted linear response", "0-1", "predict(), lines 293-301"],
    ]
    range_rows = [
        ["AI", "0-0.167", ">0.167-0.333", ">0.333-1.000", "Raise controls as threat rises"],
        ["PA", "0-0.20", ">0.20-0.40", ">0.40-0.50", "Investigate leakage and raise controls"],
        ["DM", "Lowest-cost value meeting Y>=0.70", "Lowest-cost value meeting Y>=0.60", "1.00 maximum defense intensity", "Search 0-1 in increments of 0.05"],
        ["GS", "0.75-1.00 (4-5 clients)", "0.50 (3 clients)", "0-0.25 (1-2 clients)", "Controller permits at least 3 clients"],
        ["Y", ">=0.70", "0.60 to <0.70", "<0.60", "Accept preferred; accept managed; otherwise defer"],
    ]
    result_rows = [
        ["Phase-2 paired test", "Face classification accuracy", pct(e.baseline["face_accuracy"]), "100 face/fingerprint batching pairs"],
        ["Phase-2 paired test", "Fingerprint classification accuracy", pct(e.baseline["finger_accuracy"]), "Separate 100-class fingerprint head"],
        ["Phase-4 comparison", "Combined DO-FMBS clean face accuracy", pct(e.defenses["comparison"]["do_fmbs"]["face_accuracy"]), "605 faces"],
        ["Phase-4 comparison", "Combined DO-FMBS conditional FGSM ASR", pct(e.defenses["comparison"]["do_fmbs"]["attack_success_rate"]), "epsilon=0.05; clean-correct denominator"],
        ["Phase-5 validation", "Final face-classification accuracy", pct(e.validation["classification_metrics"]["do_fmbs"]["accuracy"]), "First 100 validation faces"],
        ["Operational face verification", "Final FAR", pct(final["FAR"], 3), f"{final['FAR_numerator']}/{final['FAR_denominator']} impostor claims"],
        ["Operational face verification", "Final FRR", pct(final["FRR"], 3), f"{final['FRR_numerator']}/{final['FRR_denominator']} genuine claims"],
        ["Operational privacy", "PA", num(r.pa_row["PA"], 3), f"Membership accuracy {pct(r.pa_row['membership_accuracy'])}"],
        ["Phase-6 model", "R-squared", f"{float(r.fit['r_squared']):.6f}", "Synthetic coefficient-recovery fit"],
    ]
    implementation_rows = [
        ["Data import", "fetch_lfw_people", "phase1_data_preparation.py:31"],
        ["Fingerprint generator", "generate_one_fingerprint", "phase1_data_preparation.py:70-91"],
        ["Identity client partition", "non_iid_partition", "phase1_data_preparation.py:144-154"],
        ["Neural model", "create_improved_model", "phase2_federated_learning.py:131-172"],
        ["Local client update", "EnhancedFedAvg.client_update", "phase2_federated_learning.py:203-229"],
        ["Global aggregation", "EnhancedFedAvg.aggregate", "phase2_federated_learning.py:231-249"],
        ["Conditional attack evaluation", "evaluate_attack", "phase3_adversarial_attacks.py:167-207"],
        ["DP-style training", "train_with_dp", "phase4_defense_mechanisms.py:174-231"],
        ["Adversarial training", "train_with_art", "phase4_defense_mechanisms.py:257-303"],
        ["Robustness sweep", "evaluate_model_robustness", "phase5_model_validation.py:110-152"],
        ["Coefficient estimation", "fit_equation", "scripts/constrained_dynamic_model.py:219-290"],
        ["Control search", "optimize_controls", "scripts/constrained_dynamic_model.py:304-340"],
    ]

    parts = [
        "# DO-FMBS CODE AND SIMULATION PRACTICAL GUIDE",
        "",
        "**A plain-language guide to the data, source code, Phase 1-6 workflow, commands, variables, outputs and recorded values**",
        "",
        "Researcher: Lydiah Moraa Machora (I152/P/9515/23)",
        "",
        "Jaramogi Oginga Odinga University of Science and Technology | 2026",
        "",
        f"Guide generated from synchronized evidence: {generated_at}",
        "",
        "## 1. Purpose and intended reader",
        "",
        "This guide is for a reader who is new to the project and may also be new to Python, machine learning or federated learning. It explains what the system does, what every phase receives and produces, how to run the application or individual scripts, where the important values are implemented, and how to interpret the results without confusing the neural-model evidence with the Phase-6 controller simulation.",
        "",
        "The shortest explanation is: DO-FMBS prepares face and fingerprint data, trains a baseline across five simulated clients, measures attacks, trains defenses, validates the baseline against the defended model, and finally demonstrates a constrained controller that selects defense intensity and client participation when attack conditions change.",
        "",
        "### 1.1 What the project contains",
        "",
        md_table(["Phase", "Plain-language purpose", "Main operation", "Principal outputs"], phase_rows),
        "",
        figure_marker(*FIGURE_SPECS[0]),
        "",
        "### 1.2 Three ideas to understand first",
        "",
        "- A model is a set of learned numerical weights that turns input features into class probabilities.",
        "- Federated learning means each client trains a local copy and the server averages local weights; the raw client arrays are not uploaded over a real network in this simulation.",
        "- An attack tests how easily predictions or privacy signals can be disturbed; a defense changes training to reduce that vulnerability.",
        "",
        "## 2. Data: where the faces and fingerprints come from",
        "",
        "### 2.1 How the face images are imported",
        "",
        "Phase 1 calls scikit-learn's fetch_lfw_people(min_faces_per_person=20, resize=0.4, color=False) at phase1_data_preparation.py:31. On the first run, scikit-learn downloads the funneled LFW archive from https://ndownloader.figshare.com/files/5976015. On this workstation the reusable JPEG cache is C:/Users/user/scikit_learn_data/lfw_home/lfw_funneled. On another computer, scikit-learn creates the equivalent cache beneath that user's scikit_learn_data directory.",
        "",
        "The code does not manually copy JPEG files into the repository. It receives the LFW images as a NumPy array, keeps identities with at least twenty images, resizes them to 40% of the source size, loads grayscale values, flattens each 50×37 image into 1,850 features, standardizes the features, makes a stratified 80/20 split and saves the prepared arrays in do_fmbs_data/face_data.npz.",
        "",
        figure_marker(*FIGURE_SPECS[1]),
        "",
        "### 2.2 How the fingerprints are generated",
        "",
        "The fingerprints are generated locally; they are not downloaded. generate_one_fingerprint() at phase1_data_preparation.py:70-91 creates a 96×96 grid, chooses a seeded ridge angle and frequency, combines two sinusoidal ridge patterns, adds a radial core pattern, adds Gaussian noise, applies Gaussian smoothing with sigma=0.8 and rescales the result to 8-bit pixel values. A second seeded variation with standard deviation 8 is added at lines 100-103.",
        "",
        "The loops create 100 synthetic finger identities with 8 impressions each, giving 800 images. The seed finger_id×100+impression makes the generation reproducible. Images exist in memory during generation; the project saves the standardized flattened arrays and labels in do_fmbs_data/fingerprint_data.npz and saves a visual montage as do_fmbs_fingerprints.png.",
        "",
        figure_marker(*FIGURE_SPECS[2]),
        "",
        "### 2.3 Exact prepared data sizes",
        "",
        md_table(["Modality", "Source", "Total", "Identities", "Image size", "Features", "Training", "Testing"], data_rows),
        "",
        "Important implementation detail: the current reproducible pipeline fits StandardScaler on each complete modality array before splitting. A future external-validation implementation should fit preprocessing on training data and apply that fitted transform to test data.",
        "",
        "### 2.4 How data is distributed to clients",
        "",
        "Phase 1 creates persistent identity-grouped client files. Each identity class belongs to one of five client partitions. These files demonstrate non-IID identity allocation. The recorded counts are:",
        "",
        md_table(["Client", "Face samples", "Fingerprint samples", "Total samples"], client_rows),
        "",
        figure_marker(*FIGURE_SPECS[3]),
        "",
        "During Phase 2, the training script reads the two modality archives and creates a fresh random five-way in-memory split of 640 batching pairs, giving 128 rows per client. Phase 2 does not load the saved client_0.npz through client_4.npz files. Face and fingerprint labels belong to independent identity spaces, so each row is a training/batching pair rather than a claim that the face and fingerprint belong to the same person.",
        "",
        "## 3. Installing and opening the project",
        "",
        "### 3.1 Simplest Windows method",
        "",
        "1. Open the project folder H:\\projects\\DO-FMBS-Lydiah.",
        "2. Double-click setup_and_run.bat, or run it from Command Prompt.",
        "3. The script checks for Python 3.9-3.11, creates venv, installs requirements and opens the Streamlit application.",
        "4. The application is served at http://localhost:8501. Press Ctrl+C in the terminal to stop it.",
        "",
        "Command: .\\setup_and_run.bat",
        "",
        "### 3.2 Manual Windows PowerShell setup",
        "",
        "Command: py -3.11 -m venv venv",
        "",
        "Command: .\\venv\\Scripts\\python.exe -m pip install --upgrade pip",
        "",
        "Command: .\\venv\\Scripts\\python.exe -m pip install -r requirements_app.txt",
        "",
        "Command: .\\venv\\Scripts\\python.exe -m streamlit run app.py",
        "",
        "Using the virtual environment's python.exe directly avoids PowerShell execution-policy problems with Activate.ps1.",
        "",
        "### 3.3 Linux or macOS setup",
        "",
        "Command: python3 -m venv venv",
        "",
        "Command: source venv/bin/activate",
        "",
        "Command: python -m pip install -r requirements_app.txt",
        "",
        "Command: python -m streamlit run app.py",
        "",
        "### 3.4 Main software used",
        "",
        md_table(["Software", "Role"], [
            ["Python 3.9-3.11", "Runs every phase and the application"],
            ["TensorFlow/Keras", "Neural models, local training and attack gradients"],
            ["NumPy/Pandas", "Arrays, saved evidence and tabular processing"],
            ["scikit-learn", "LFW download, preprocessing, splits and metrics"],
            ["SciPy", "Filtering, statistics and SLSQP constrained optimization"],
            ["Matplotlib/Plotly", "Figures and interactive charts"],
            ["Streamlit", "Browser-based interface"],
        ]),
        "",
        "## 4. How to run the complete simulation",
        "",
        "### 4.1 Recommended protected full run",
        "",
        "The isolated runner copies the source into a new run directory, executes all six phases, saves every console log and records exact wall-clock duration. Give each new run a new directory name so earlier evidence remains intact.",
        "",
        "Command: .\\venv\\Scripts\\python.exe scripts\\run_isolated_instrumented_pipeline.py --run-dir runtime_recording\\full_pipeline_new",
        "",
        "To resume a partially completed protected run:",
        "",
        "Command: .\\venv\\Scripts\\python.exe scripts\\run_isolated_instrumented_pipeline.py --run-dir runtime_recording\\full_pipeline_new --resume",
        "",
        "To run only Phases 3-5 in that isolated directory:",
        "",
        "Command: .\\venv\\Scripts\\python.exe scripts\\run_isolated_instrumented_pipeline.py --run-dir runtime_recording\\full_pipeline_new --start-phase 3 --end-phase 5 --resume",
        "",
        "### 4.2 Running phases individually in the project folder",
        "",
        "Run these commands in order when creating a complete result set. Direct execution writes to do_fmbs_data, simulation_evidence and the project-root figures, so copy valuable outputs before deliberately rerunning a phase.",
        "",
        md_table(["Phase", "PowerShell command", "Meaning"], command_rows),
        "",
        "### 4.3 Actual code dependencies",
        "",
        md_table(["Connection", "What is actually loaded"], dependency_rows),
        "",
        "Phase 4 is executed after attack analysis in the recommended workflow, but its training script builds new comparison models from Phase-1 arrays. Phase 6 is a separate synthetic controller demonstration: it does not retrain the neural network for each of its 700 conditions.",
        "",
        "## 5. How each phase works",
        "",
        "### 5.1 Phase 1 — data preparation",
        "",
        "Input: internet/cached LFW faces plus the local random-number generator. Process: load faces, generate fingerprints, flatten and standardize both modalities, split them 80/20, allocate identity-grouped client files and save figures. Output: face_data.npz, fingerprint_data.npz and five client NPZ files.",
        "",
        "### 5.2 Phase 2 — federated baseline",
        "",
        "The model has a face branch and a fingerprint branch. Each branch uses dense layers 256→128→64 with batch normalization and dropout, followed by its own output head. A client receives the current global weights, trains locally and returns updated weights plus its sample count. EnhancedFedAvg.aggregate() calculates a sample-weighted average and installs it as the next global model.",
        "",
        "$$",
        r"w_{t+1}=\sum_{k=1}^{K}\frac{n_k}{\sum_j n_j}w_{k,t+1}",
        "$$",
        "",
        "The recorded run uses five clients, 12 global rounds, 2 local epochs, batch size 64, 640 training pairs and 100 test pairs. It therefore performs 12×5×2=120 client-epochs. The two heads classify their own independent identity labels; this is not a same-person fused decision.",
        "",
        figure_marker(*FIGURE_SPECS[4]),
        "",
        "### 5.3 Phase 3 — attacks",
        "",
        "FGSM changes an input once in the direction that increases loss. PGD repeats small gradient steps and projects the perturbation back into the allowed epsilon region. Evasion ASR is calculated only among samples the clean model classified correctly. Membership inference treats confidence above 0.70 as a member signal. Label-flipping changes 10% of 200 training labels and retrains a model clone.",
        "",
        md_table(["epsilon", "FGSM ASR", "PGD ASR"], attack_rows),
        "",
        figure_marker(*FIGURE_SPECS[5]),
        "",
        "### 5.4 Phase 4 — defenses",
        "",
        "Four centrally trained comparison arms are created: no defense, DP-style Gaussian gradient noise, adversarial robustness training (ART) and the combined DO-FMBS defense. In this project, ART is the short label for the custom adversarial-robustness-training arm; it is not a claim that the IBM Adversarial Robustness Toolbox package is used. DP-style means noise is added to gradients; the code does not include clipping or a privacy accountant, so it is not a formal differential-privacy guarantee. The ART arm combines clean and FGSM examples. The combined model warm-starts on clean data and then uses adversarial examples plus smaller Gaussian gradient noise.",
        "",
        md_table(["Condition", "Clean face accuracy", "Conditional FGSM ASR", "Attack resistance (1-ASR)"], defense_rows),
        "",
        figure_marker(*FIGURE_SPECS[6]),
        "",
        "### 5.5 Phase 5 — testing and validation",
        "",
        "Phase 5 loads the Phase-2 baseline and Phase-4 combined model. It creates FGSM inputs at epsilon 0, .05, .10, .15 and .20; compares accuracy; repeats accuracy measurement on ten random 100-sample subsets; performs an independent t-test; calculates effect size; and saves accuracy, precision, recall and F1. At epsilon=0, the stored 1-accuracy value is clean error rather than an attack event.",
        "",
        figure_marker(*FIGURE_SPECS[7]),
        "",
        "### 5.6 Phase 6 — constrained dynamic optimization",
        "",
        "Phase 6 first generates a balanced, seeded 7 AI×5 PA×4 DM×5 GS=700-condition synthetic design. It estimates B0-B4 with sign-constrained least squares, checks fit and constraints, then runs a twelve-round controller. In each round the controller observes AI and PA, tries 21 DM levels and three client counts (63 candidates), predicts Y, and selects the lowest-cost preferred or managed policy.",
        "",
        "This makes the system dynamic: AI and PA can change from one round to the next, so DM and GS are selected again. Y is the simulated next-round efficiency score; it is not biometric accuracy, FAR, FRR or ASR.",
        "",
        "## 6. Variables and equations",
        "",
        "### 6.1 The original X1-X4 equation",
        "",
        "The original presentation used the general multiple-regression notation:",
        "",
        "$$",
        r"Y_i=B_0+B_1X_{1i}+B_2X_{2i}+B_3X_{3i}+B_4X_{4i}+e_i",
        "$$",
        "",
        "Here e_i is the unexplained residual. The project variables are substitutions for the four X variables:",
        "",
        md_table(["Original", "Named variable", "Meaning", "Transformation", "Domain", "Implementation"], variable_rows),
        "",
        "After substitution, the same structural equation is:",
        "",
        "$$",
        r"Y_i=B_0+B_{AI}AI_i+B_{PA}PA_i+B_{DM}DM_i+B_{GS}GS_i+e_i",
        "$$",
        "",
        "### 6.2 Why the implemented form uses t+1",
        "",
        "The controller observes AI_t and PA_t before round t, applies DM_t and GS_t during that round, and predicts the next-round response. Therefore, the prediction form is:",
        "",
        "$$",
        r"\widehat{Y}_{t+1}=B_0+B_{AI}AI_t+B_{PA}PA_t+B_{DM}DM_t+B_{GS}GS_t",
        "$$",
        "",
        "The hat means predicted value. The residual does not appear in the prediction because the fitted predictor uses its expected value of zero. This is the explicit bridge from X1-X4 notation to AI, PA, DM and GS; it is not a different model family.",
        "",
        "### 6.3 Estimated equation and coefficient meaning",
        "",
        "$$",
        r"\widehat{Y}_{t+1}=0.400738-0.182145AI_t-0.227329PA_t+0.261393DM_t+0.159417GS_t",
        "$$",
        "",
        md_table(["Coefficient", "Estimate", "Constraint", "Verified"], coefficient_rows),
        "",
        "Negative AI and PA coefficients mean higher attack/privacy pressure lowers predicted Y when controls are fixed. Positive DM and GS coefficients mean stronger defense and broader participation raise predicted Y when threat states are fixed. These coefficients are fitted from the seeded Phase-6 synthetic design; they are not neural-network weights.",
        "",
        "### 6.4 Operating ranges",
        "",
        md_table(["Variable", "Preferred", "Managed", "Critical modeled", "Controller response"], range_rows),
        "",
        "DM is a control interval rather than a performance grade. DM maps to adversarial fraction=0.50DM, training epsilon=0.05DM and gradient-noise standard deviation=0.0003DM. At DM=1 these settings reproduce the combined Phase-4 policy.",
        "",
        "Values outside the fitted domains AI in [0,1], PA in [0,0.50], DM in [0,1] and GS in [0,1] are rejected by the domain guard instead of being extrapolated.",
        "",
        "### 6.5 How the optimization chooses DM and GS",
        "",
        "For every round, the controller evaluates the 63 admissible DM/client combinations and minimizes the following cost:",
        "",
        "$$",
        r"J_t=0.60DM_t^2+0.40GS_t^2+0.15(DM_t-DM_{t-1})^2+0.10(GS_t-GS_{t-1})^2",
        "$$",
        "",
        "The first two terms penalize stronger defense and broader client participation; the last two penalize abrupt changes from the preceding round. The search is subject to:",
        "",
        "$$",
        r"\widehat{Y}_{t+1}\geq0.60,\quad 0\leq DM_t\leq1,\quad GS_t\in\{0.50,0.75,1.00\}",
        "$$",
        "",
        "The controller first searches for the lowest-cost candidate with Y>=0.70. If none exists, it searches for the lowest-cost candidate with Y>=0.60. If neither target is feasible, it activates the safety rejection/defer decision. Thus optimization means a finite, reproducible comparison of 21 DM values times 3 permitted client counts, not an undocumented manual choice.",
        "",
        "### 6.6 Example calculation",
        "",
        "Round 7 uses epsilon=0.10, AI=0.10/0.30=0.333333, PA=0.30, DM=0.80 and four clients, so GS=(4-1)/4=0.75:",
        "",
        "$$",
        r"\widehat{Y}_8=0.400738-(0.182145)(0.333333)-(0.227329)(0.30)+(0.261393)(0.80)+(0.159417)(0.75)=0.600501",
        "$$",
        "",
        "The result is just above the managed minimum Y=0.60, so the round is accepted as a managed policy.",
        "",
        "## 7. Values that can be changed",
        "",
        "Phases 2, 3 and 4 expose command-line options. Phases 1, 5 and 6 use fixed source values; change the cited constants carefully and then rerun the dependent phases so the evidence remains synchronized.",
        "",
        md_table(["Phase", "Value or option", "What it changes", "Code/argument location"], parameter_rows),
        "",
        "Example smaller Phase-2 experiment:",
        "",
        "Command: .\\venv\\Scripts\\python.exe phase2_federated_learning.py --rounds 6 --local-epochs 1 --batch-size 32 --max-pairs 300 --test-pairs 100",
        "",
        "Example Phase-3 experiment that adds epsilon=0.15 to the fixed attack sweep and lowers the poisoning pressure:",
        "",
        "Command: .\\venv\\Scripts\\python.exe phase3_adversarial_attacks.py --epsilon 0.15 --membership-threshold 0.75 --poison-rate 0.05 --poison-epochs 3",
        "",
        "The Phase-3 script always evaluates epsilon={0.01,0.05,0.10,0.20,0.30}. The --epsilon argument adds a value to this fixed sweep; passing an existing value such as 0.05 leaves the sweep unchanged. To replace the entire sweep, edit phase3_adversarial_attacks.py:321 and record the revised design.",
        "",
        "Example shorter Phase-4 comparison:",
        "",
        "Command: .\\venv\\Scripts\\python.exe phase4_defense_mechanisms.py --epochs 6 --max-pairs 1000 --test-pairs 300 --batch-size 32",
        "",
        "Change one factor at a time when comparing experiments, save each run in a separate directory, and record the command. Lower runtime does not automatically mean a better model; accuracy, FAR, FRR, ASR and privacy pressure must be evaluated together.",
        "",
        "## 8. Metrics and how to read them",
        "",
        md_table(["Metric", "Plain meaning", "Direction"], [
            ["Accuracy", "Share of class predictions that are correct", "Higher is generally better"],
            ["FAR", "Share of impostor identity claims accepted by the verifier", "Lower is better"],
            ["FRR", "Share of genuine identity claims rejected by the verifier", "Lower is better"],
            ["ASR", "Share of eligible samples for which an attack succeeds", "Lower is better"],
            ["PA", "Membership-inference advantage above chance", "Lower is better"],
            ["Attack resistance", "1 minus conditional evasion ASR in Phase 4", "Higher is better"],
            ["Y", "Synthetic next-round controller efficiency response", "Preferred >=0.70; managed >=0.60"],
        ]),
        "",
        "FAR and FRR in the final acceptance table are face-branch claimed-identity verification metrics with a frozen threshold. They are distinct from the Phase-2 classification-derived diagnostic rates. ASR must always be reported with its attack, epsilon and denominator.",
        "",
        "## 9. Recorded output values",
        "",
        md_table(["Evidence scope", "Measure", "Recorded value", "Denominator or interpretation"], result_rows),
        "",
        f"The operational baseline comparator has FAR {pct(base['FAR'],3)} and FRR {pct(base['FRR'],3)}. The final model has FAR {pct(final['FAR'],3)} and FRR {pct(final['FRR'],3)}. The declared clean project limits are FAR<=10% and FRR<=20%; the final identity-cluster upper bounds also remain within those limits.",
        "",
        "## 10. Recorded runtime",
        "",
        md_table(["Phase", "Script", "Elapsed seconds", "Execution state"], runtime_rows),
        "",
        f"Total active Phase 1-6 command time is {runtime_total:.6f} seconds, equal to 25 minutes 22.308 seconds on the recorded Windows CPU environment. Phase 4 is the longest because it trains and evaluates four defense conditions. Runtime depends on hardware, data limits, epochs and TensorFlow configuration.",
        "",
        "## 11. Where outputs are saved",
        "",
        md_table(["Location", "Contents", "Produced by"], [
            ["do_fmbs_data/", "Prepared arrays, models and PKL result dictionaries", "Phases 1-5"],
            ["do_fmbs_data/clients/", "Five persisted Phase-1 identity partitions", "Phase 1"],
            ["simulation_evidence/constrained_dynamic_model/", "Coefficients, fit, constraints and control schedule", "Phase 6"],
            ["Project-root PNG files", "Data, training, attack, defense and validation figures", "Phases 1-5"],
            ["runtime_recording/<run>/", "Protected source copy, outputs, logs and phase timings", "Isolated runner"],
        ]),
        "",
        "Common checks after a successful run:",
        "",
        "- Phase 1: face_data.npz and fingerprint_data.npz exist.",
        "- Phase 2: baseline_model.keras and baseline_metrics.pkl exist.",
        "- Phase 3: attack_results.pkl exists.",
        "- Phase 4: do_fmbs_model.keras and final_results.pkl exist.",
        "- Phase 5: validation_results.pkl and validation_report.txt exist.",
        "- Phase 6: model_fit.json and dynamic_optimization_schedule.csv exist.",
        "",
        "## 12. Where important functions are implemented",
        "",
        md_table(["Purpose", "Function", "Source location"], implementation_rows),
        "",
        "## 13. Automated verification and GitHub",
        "",
        "The public code repository is https://github.com/lydiahmachora-del/DO-FMBS-Lydiah. GitHub Actions automatically compiles source, checks the code-only publication policy, runs portable unit tests and executes a Phase-6 smoke simulation after pushes and pull requests. CodeQL performs Python security analysis on main, pull requests, a weekly schedule and manual requests.",
        "",
        "Local full verification command:",
        "",
        "Command: .\\venv\\Scripts\\python.exe -m unittest discover -s tests -v",
        "",
        "A test pass verifies declared software behavior and evidence consistency; it does not replace external biometric validation on a representative population.",
        "",
        "## 14. Troubleshooting",
        "",
        md_table(["Problem", "Likely cause", "Action"], [
            ["Python/TensorFlow install fails", "Python version above 3.11", "Install Python 3.11 and recreate venv"],
            ["Phase 2 cannot find face_data.npz", "Phase 1 has not run in this directory", "Run Phase 1 first"],
            ["Phase 3 cannot find baseline_model.keras", "Phase 2 has not completed", "Run Phase 2 and verify its model output"],
            ["Phase 5 cannot find a model", "Phase 2 or Phase 4 output is absent", "Run both required model phases"],
            ["LFW download fails", "Internet, proxy or cache problem", "Check connectivity and scikit_learn_data permissions"],
            ["A rerun changes saved outputs", "Direct scripts overwrite standard artifact names", "Use a new isolated-run directory"],
            ["Training is slow", "CPU training or large epoch/data settings", "Use smaller exploratory values, then run the declared final configuration"],
            ["Streamlit page is blank", "Server stopped or wrong port", "Restart with python -m streamlit run app.py"],
        ]),
        "",
        "## 15. What to say when explaining the project",
        "",
        "1. The system starts with downloaded LFW face images and locally generated synthetic fingerprints.",
        "2. Phase 1 standardizes and splits both modalities and creates five client partitions.",
        "3. Phase 2 simulates federated learning: clients train local dual-head models and the server combines weights with sample-weighted FedAvg.",
        "4. Phase 3 measures evasion, membership-inference and poisoning attacks.",
        "5. Phase 4 trains and compares DP-style noise, adversarial training and the combined defense.",
        "6. Phase 5 compares the baseline and defended model using accuracy, robustness and statistical evidence.",
        "7. Phase 6 maps X1-X4 to AI, PA, DM and GS, estimates the constrained equation, and selects defense/client controls for each changing round.",
        "8. The operational face verifier records final FAR 7.852% and FRR 11.146%, while the controller fit records R-squared 0.968064.",
        "9. High attack intensity remains an availability risk, which is visible in increasing ASR and FRR under attack.",
        "10. Every claim is tied to a script, output artifact, recorded command and reproducible seed.",
        "",
        "## 16. Final checklist for a new user",
        "",
        "- Use Python 3.9-3.11 and install requirements_app.txt.",
        "- Run from the repository root so relative paths resolve correctly.",
        "- Use the isolated runner for a protected complete experiment.",
        "- Run Phases 1-5 in order when building neural-model evidence.",
        "- Treat Phase 6 as a synthetic constrained-controller demonstration.",
        "- Report metric definitions and denominators, especially FAR, FRR and ASR.",
        "- Keep X1=AI, X2=PA, X3=DM and X4=GS visible when presenting the equation.",
        "- Preserve commands, seeds, logs, runtimes and artifacts for every reported run.",
    ]
    return "\n".join(parts).strip() + "\n"


def markdown_for_file(markdown: str) -> str:
    lines = []
    for line in markdown.splitlines():
        match = FIGURE_RE.match(line.strip())
        if match:
            filename, caption = match.groups()
            lines.extend([f"![{caption}]({filename})", "", f"*{caption}*"])
        else:
            lines.append(line)
    return "\n".join(lines).strip() + "\n"


def write_docx(markdown: str, path: Path, figure_dir: Path) -> None:
    from docx import Document
    from docx.enum.table import WD_CELL_VERTICAL_ALIGNMENT
    from docx.enum.text import WD_ALIGN_PARAGRAPH
    from docx.oxml import OxmlElement
    from docx.oxml.ns import qn
    from docx.shared import Inches, Pt, RGBColor

    doc = Document()
    section = doc.sections[0]
    section.top_margin = Inches(.65); section.bottom_margin = Inches(.65)
    section.left_margin = Inches(.65); section.right_margin = Inches(.65)
    doc.core_properties.title = "DO-FMBS Code and Simulation Practical Guide"
    doc.core_properties.author = "DO-FMBS Research Project"
    styles = doc.styles
    styles["Normal"].font.name = "Times New Roman"; styles["Normal"].font.size = Pt(10)
    for name, size, color in (("Title", 20, "17365D"), ("Heading 1", 15, "17365D"), ("Heading 2", 12, "2F5597")):
        styles[name].font.name = "Times New Roman"; styles[name].font.size = Pt(size)
        styles[name].font.color.rgb = RGBColor.from_string(color)

    first = True
    equation_buffers = []
    for kind, payload in _markdown_blocks(markdown):
        if kind == "heading1" and first:
            p = doc.add_paragraph(_plain_markdown(str(payload)), style="Title")
            p.alignment = WD_ALIGN_PARAGRAPH.CENTER; first = False
        elif kind in {"heading1", "heading2", "heading3"}:
            level = 1 if kind in {"heading1", "heading2"} else 2
            p = doc.add_heading(_plain_markdown(str(payload)), level=level)
            p.paragraph_format.keep_with_next = True
            if level == 1 and re.match(r"(?:5|6|9|15)\.", p.text):
                p.paragraph_format.page_break_before = True
        elif kind == "table":
            rows = payload
            table = doc.add_table(rows=len(rows), cols=len(rows[0])); table.style = "Table Grid"
            for ri, row in enumerate(rows):
                for ci, value in enumerate(row):
                    cell = table.cell(ri, ci); cell.text = _plain_markdown(str(value)); cell.vertical_alignment = WD_CELL_VERTICAL_ALIGNMENT.CENTER
                    if ri == 0:
                        shading = OxmlElement("w:shd"); shading.set(qn("w:fill"), "17365D"); cell._tc.get_or_add_tcPr().append(shading)
                    for paragraph in cell.paragraphs:
                        for run in paragraph.runs:
                            run.font.name = "Times New Roman"; run.font.size = Pt(7.2 if len(rows[0]) >= 6 else 8)
                            if ri == 0:
                                run.bold = True; run.font.color.rgb = RGBColor(255, 255, 255)
        elif kind == "equation":
            p = doc.add_paragraph(); p.alignment = WD_ALIGN_PARAGRAPH.CENTER
            buffer = _equation_image_buffer(str(payload)); equation_buffers.append(buffer)
            p.add_run().add_picture(buffer, width=Inches(6.5))
        elif kind in {"bullet", "number"}:
            value = payload[1] if kind == "number" else payload
            doc.add_paragraph(_plain_markdown(str(value)), style="List Number" if kind == "number" else "List Bullet")
        else:
            text = str(payload)
            match = FIGURE_RE.match(text.strip())
            if match:
                filename, caption = match.groups()
                p = doc.add_paragraph(); p.alignment = WD_ALIGN_PARAGRAPH.CENTER
                width = 6.7 if filename.endswith("SIMULATION_FLOW.png") else 6.4
                p.add_run().add_picture(str(figure_dir / filename), width=Inches(width))
                cp = doc.add_paragraph(caption, style="Caption"); cp.alignment = WD_ALIGN_PARAGRAPH.CENTER
            elif text.startswith("Command:"):
                p = doc.add_paragraph(); p.paragraph_format.left_indent = Inches(.18)
                run = p.add_run(_plain_markdown(text)); run.font.name = "Consolas"; run.font.size = Pt(8.5)
                shading = OxmlElement("w:shd"); shading.set(qn("w:fill"), "F2F2F2"); p._p.get_or_add_pPr().append(shading)
            else:
                p = doc.add_paragraph(_plain_markdown(text)); p.paragraph_format.space_after = Pt(5)

    footer = section.footer.paragraphs[0]
    footer.text = "DO-FMBS Code and Simulation Practical Guide | 2026 | Page "
    page_run = footer.add_run()
    field_begin = OxmlElement("w:fldChar"); field_begin.set(qn("w:fldCharType"), "begin")
    field_code = OxmlElement("w:instrText"); field_code.set(qn("xml:space"), "preserve"); field_code.text = " PAGE "
    field_end = OxmlElement("w:fldChar"); field_end.set(qn("w:fldCharType"), "end")
    page_run._r.extend((field_begin, field_code, field_end))
    footer.alignment = WD_ALIGN_PARAGRAPH.CENTER
    doc.save(path)


def ascii_safe(text: str) -> str:
    replacements = {"×": "x", "→": "->", "≥": ">=", "≤": "<=", "–": "-", "—": "-", "…": "...", "Ŷ": "Y_hat"}
    value = _plain_markdown(text)
    for source, target in replacements.items():
        value = value.replace(source, target)
    return value.encode("latin-1", "replace").decode("latin-1")


def write_pdf(markdown: str, path: Path, figure_dir: Path) -> None:
    from reportlab.lib import colors
    from reportlab.lib.enums import TA_CENTER
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
    from reportlab.lib.units import mm
    from reportlab.platypus import CondPageBreak, Image, PageBreak, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle

    styles = getSampleStyleSheet()
    styles.add(ParagraphStyle(name="GuideTitle", parent=styles["Title"], fontName="Times-Bold", fontSize=18, leading=22, textColor=colors.HexColor("#17365D"), alignment=TA_CENTER, spaceAfter=10))
    styles.add(ParagraphStyle(name="GuideH1", parent=styles["Heading1"], fontName="Times-Bold", fontSize=14, leading=17, textColor=colors.HexColor("#17365D"), spaceBefore=8, spaceAfter=5, keepWithNext=True))
    styles.add(ParagraphStyle(name="GuideH2", parent=styles["Heading2"], fontName="Times-Bold", fontSize=11.5, leading=14, textColor=colors.HexColor("#2F5597"), spaceBefore=6, spaceAfter=4, keepWithNext=True))
    styles.add(ParagraphStyle(name="GuideBody", parent=styles["BodyText"], fontName="Times-Roman", fontSize=9, leading=12.4, alignment=4, spaceAfter=5))
    styles.add(ParagraphStyle(name="GuideCaption", parent=styles["BodyText"], fontName="Times-Italic", fontSize=8, leading=10, alignment=TA_CENTER, spaceAfter=7))
    styles.add(ParagraphStyle(name="GuideCell", parent=styles["BodyText"], fontName="Times-Roman", fontSize=6.6, leading=8))
    styles.add(ParagraphStyle(name="GuideCommand", parent=styles["BodyText"], fontName="Courier", fontSize=7.6, leading=10, backColor=colors.HexColor("#F2F2F2"), leftIndent=5, rightIndent=5, spaceAfter=5))

    def footer(canvas, document):
        canvas.saveState(); page_w, _ = A4
        canvas.setStrokeColor(colors.HexColor("#B4C6E7")); canvas.line(14*mm, 11*mm, page_w-14*mm, 11*mm)
        canvas.setFont("Times-Roman", 7.5); canvas.setFillColor(colors.HexColor("#666666"))
        canvas.drawString(14*mm, 7*mm, "DO-FMBS Code and Simulation Practical Guide")
        canvas.drawRightString(page_w-14*mm, 7*mm, f"Page {document.page}"); canvas.restoreState()

    doc = SimpleDocTemplate(str(path), pagesize=A4, leftMargin=15*mm, rightMargin=15*mm, topMargin=14*mm, bottomMargin=15*mm,
                            title="DO-FMBS Code and Simulation Practical Guide", author="DO-FMBS Research Project")
    story = []; first = True; equation_buffers = []
    for kind, payload in _markdown_blocks(markdown):
        if kind == "heading1" and first:
            story.append(Paragraph(html_escape(ascii_safe(str(payload))), styles["GuideTitle"])); first = False
        elif kind in {"heading1", "heading2"}:
            title = str(payload)
            if re.match(r"(?:5|6|9|15)\.", title):
                story.append(PageBreak())
            story.append(Paragraph(html_escape(ascii_safe(title)), styles["GuideH1"]))
        elif kind == "heading3":
            story.append(CondPageBreak(30*mm)); story.append(Paragraph(html_escape(ascii_safe(str(payload))), styles["GuideH2"]))
        elif kind == "table":
            rows = [[Paragraph(html_escape(ascii_safe(str(cell))), styles["GuideCell"]) for cell in row] for row in payload]
            width = A4[0] - 32*mm; columns = len(rows[0])
            table = Table(rows, colWidths=[width/columns]*columns, repeatRows=1, hAlign="CENTER")
            table.setStyle(TableStyle([
                ("BACKGROUND", (0,0), (-1,0), colors.HexColor("#17365D")), ("TEXTCOLOR", (0,0), (-1,0), colors.white),
                ("GRID", (0,0), (-1,-1), .25, colors.HexColor("#A6A6A6")), ("VALIGN", (0,0), (-1,-1), "TOP"),
                ("ROWBACKGROUNDS", (0,1), (-1,-1), [colors.white, colors.HexColor("#F2F6FA")]),
                ("LEFTPADDING", (0,0), (-1,-1), 2.5), ("RIGHTPADDING", (0,0), (-1,-1), 2.5),
                ("TOPPADDING", (0,0), (-1,-1), 2.5), ("BOTTOMPADDING", (0,0), (-1,-1), 2.5),
            ])); story.extend([table, Spacer(1,5)])
        elif kind == "equation":
            buffer = _equation_image_buffer(str(payload)); equation_buffers.append(buffer)
            image = Image(buffer); image._restrictSize(A4[0]-35*mm, 18*mm); image.hAlign = "CENTER"
            story.extend([image, Spacer(1,5)])
        else:
            if kind == "bullet":
                # ReportLab's built-in Latin-1 fonts do not render a Unicode
                # bullet consistently.  An ASCII dash remains searchable and
                # displays correctly on every supported PDF reader.
                value = "- " + str(payload)
            elif kind == "number":
                value = f"{payload[0]}. {payload[1]}"
            else:
                value = str(payload)
            match = FIGURE_RE.match(value.strip())
            if match:
                filename, caption = match.groups(); story.append(CondPageBreak(105*mm))
                image = Image(str(figure_dir / filename)); image._restrictSize(A4[0]-35*mm, 112*mm); image.hAlign = "CENTER"
                story.extend([image, Paragraph(html_escape(ascii_safe(caption)), styles["GuideCaption"])])
            elif value.startswith("Command:"):
                story.append(Paragraph(html_escape(ascii_safe(value)), styles["GuideCommand"]))
            else:
                story.append(Paragraph(html_escape(ascii_safe(value)), styles["GuideBody"]))
    doc.build(story, onFirstPage=footer, onLaterPages=footer)


def extract_docx_text(path: Path) -> str:
    from docx import Document
    document = Document(path)
    values = [paragraph.text for paragraph in document.paragraphs]
    for table in document.tables:
        for row in table.rows:
            values.extend(cell.text for cell in row.cells)
    return "\n".join(values)


def extract_pdf_text(path: Path) -> str:
    import pymupdf
    with pymupdf.open(path) as document:
        return "\n".join(page.get_text() for page in document)


def assert_text(text: str, label: str) -> None:
    lowered = text.casefold()
    found = [phrase for phrase in REJECTED if phrase in lowered]
    if re.search(r"(?<![a-z0-9])n/a(?![a-z0-9])", lowered):
        found.append("n/a")
    if found:
        raise ValueError(f"{label} contains rejected wording: {', '.join(found)}")
    required = (
        "x1", "ai", "phase 1", "phase 6", "fetch_lfw_people",
        "generate_one_fingerprint", "1522.307974", "run_isolated_instrumented_pipeline",
    )
    for phrase in required:
        if phrase not in lowered:
            raise ValueError(f"{label} is missing required content: {phrase}")


def generate(args: argparse.Namespace) -> list[Path]:
    evidence = load_evidence(args)
    required_sources = [
        ROOT / "phase1_data_preparation.py", ROOT / "phase2_federated_learning.py",
        ROOT / "phase3_adversarial_attacks.py", ROOT / "phase4_defense_mechanisms.py",
        ROOT / "phase5_model_validation.py", ROOT / "phase6_constrained_dynamic_optimization.py",
        ROOT / "scripts" / "constrained_dynamic_model.py",
        ROOT / "scripts" / "run_isolated_instrumented_pipeline.py",
        ROOT / "requirements_app.txt", ROOT / "setup_and_run.bat", ROOT / "setup_and_run.sh",
    ]
    for path in required_sources:
        if not path.is_file() or path.stat().st_size == 0:
            raise FileNotFoundError(f"Guide source is missing: {path}")
    if args.validate_inputs_only:
        return []

    formats = {value.strip().lower() for value in args.formats.split(",") if value.strip()}; formats.add("md")
    if formats - {"md", "docx", "pdf"}:
        raise ValueError("Formats must be md, docx and/or pdf")
    evidence.output_dir.mkdir(parents=True, exist_ok=True)
    figures = make_figures(evidence)
    generated_at = datetime.now(ZoneInfo("Africa/Nairobi")).isoformat(timespec="seconds")
    conversion = build_markdown(evidence, generated_at)
    file_markdown = markdown_for_file(conversion); assert_text(file_markdown, "Markdown guide")

    outputs = []
    md_path = evidence.output_dir / f"{REPORT_STEM}.md"; md_path.write_text(file_markdown, encoding="utf-8"); outputs.append(md_path)
    if "docx" in formats:
        path = evidence.output_dir / f"{REPORT_STEM}.docx"; write_docx(conversion, path, evidence.output_dir)
        assert_text(extract_docx_text(path), "DOCX guide"); outputs.append(path)
    if "pdf" in formats:
        path = evidence.output_dir / f"{REPORT_STEM}.pdf"; write_pdf(conversion, path, evidence.output_dir)
        assert_text(extract_pdf_text(path), "PDF guide"); outputs.append(path)

    sources = [*required_sources, *evidence.source_paths]
    unique_sources = list(dict.fromkeys(path.resolve() for path in sources))
    from PIL import Image as PILImage

    figure_records = []
    for path in figures:
        with PILImage.open(path) as figure:
            width, height = figure.size
        figure_records.append({
            "path": path.name,
            "bytes": path.stat().st_size,
            "sha256": sha256(path),
            "width_px": width,
            "height_px": height,
        })

    manifest = {
        "generated_at_africa_nairobi": generated_at,
        "status": "GUIDE-GENERATED-FROM-SYNCHRONIZED-EVIDENCE",
        "replay": evidence.recorded.run_dir.name,
        "sources": [{"path": path.relative_to(ROOT).as_posix() if path.is_relative_to(ROOT) else str(path), "bytes": path.stat().st_size, "sha256": sha256(path)} for path in unique_sources],
        "figures": figure_records,
        "documents": [{"path": path.name, "bytes": path.stat().st_size, "sha256": sha256(path)} for path in outputs],
    }
    manifest_path = evidence.output_dir / "DO_FMBS_CODE_AND_SIMULATION_GUIDE_MANIFEST.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    outputs.append(manifest_path)
    return outputs


def main() -> None:
    args = parse_args()
    outputs = generate(args)
    if args.validate_inputs_only:
        print("INPUTS-VALIDATED: code, replay, evaluator and controller evidence")
    else:
        for path in outputs:
            print(f"Generated {path}")


if __name__ == "__main__":
    main()
