"""Generate the professional DO-FMBS response-to-examiners package.

This report is separate from the thesis chapter and every earlier project
document.  Numerical statements are loaded from the synchronized Phase 1-6
replay and the operational claimed-identity evaluator.  The report answers all
17 mock-presentation comments, distinguishes completed evidence from proposed
manuscript corrections, and states the experimental boundaries explicitly.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
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
)
from scripts.generate_constrained_dynamic_report import (
    _equation_image_buffer,
    _markdown_blocks,
    _plain_markdown,
)


REPORT_STEM = "DO_FMBS_RESPONSE_TO_MOCK_PRESENTATION_COMMENTS"
DEFAULT_OUTPUT_DIR = ROOT / "docs" / "examiner_response"
FIGURE_NAME = "EXAMINER_RESPONSE_FIGURE_1_EVIDENCE_AND_DYNAMIC_CONTROL.png"
FIGURE_RE = re.compile(r"^\[\[FIGURE:([^|]+)\|(.+)\]\]$")
URL_RE = re.compile(r"https?://[^\s<>]+")


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate the DO-FMBS response-to-examiners report.")
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


def percent(value: object, digits: int = 2) -> str:
    return f"{100.0 * float(value):.{digits}f}%"


def number(value: object, digits: int = 6) -> str:
    return f"{float(value):.{digits}f}"


def equation(latex: str) -> list[str]:
    return ["$$", latex, "$$", ""]


def response_blocks(
    number_value: int,
    title: str,
    concern: str,
    direct: list[str],
    evidence: list[str],
    correction: list[str],
    oral: str,
) -> list[str]:
    def separated(items: list[str]) -> list[str]:
        """Keep every response item as its own Markdown block.

        This is especially important for figure markers and display equations:
        without an empty line, the limited Markdown parser merges a marker into
        the preceding prose paragraph and the document converters cannot render
        it as a figure.
        """
        result: list[str] = []
        inside_equation = False
        for item in items:
            if item == "$$":
                result.append(item)
                inside_equation = not inside_equation
                if not inside_equation:
                    result.append("")
            elif inside_equation:
                result.append(item)
            elif item:
                result.extend([item, ""])
            elif not result or result[-1] != "":
                result.append("")
        return result

    values = [
        f"## Response to Comment {number_value}: {title}",
        "",
        "### A. Examiner's concern",
        "",
        concern,
        "",
        "### B. Direct response",
        "",
        *separated(direct),
        "### C. Evidence and computation",
        "",
        *separated(evidence),
        "### D. Manuscript correction and disposition",
        "",
        *separated(correction),
        "### E. Concise oral-defense answer",
        "",
        f"**Say:** {oral}",
        "",
    ]
    return values


def build_markdown(evidence, generated_at: str) -> str:
    recorded = evidence.recorded
    final = next(row for row in recorded.clean_rows if row["model"] == "DO-FMBS final")
    comparator = next(row for row in recorded.clean_rows if row["model"] == "Baseline comparator")
    fit = recorded.fit
    coefficient_map = {row["parameter"]: row for row in evidence.coefficients}
    total_runtime = sum(float(row["elapsed_seconds"]) for row in recorded.runtime_rows)
    preferred_rounds = sum(row["status"] == "preferred_target_met" for row in recorded.schedule_rows)
    managed_rounds = sum(row["status"] == "minimum_target_met" for row in recorded.schedule_rows)

    status_rows = [
        [1, "Variable values, bands and out-of-domain policy", "Evidence-ready", "Insert range table and rejection rule"],
        [2, "Variable/equation alignment", "Evidence-ready", "Use X1=AI, X2=PA, X3=DM and X4=GS consistently"],
        [3, "Dynamic mechanism and optimization", "Evidence-ready", "Distinguish coefficient fitting from round control"],
        [4, "Federated optimization values", "Evidence-ready", "Separate fixed hyperparameters from optimized controls"],
        [5, "Architecture, data source and scientific method", "Evidence-ready", "State simulation boundaries in architecture section"],
        [6, "Face/fingerprint rationale and extensibility", "Answer-ready; extension remains future work", "Do not claim proven multimodal gain"],
        [7, "Most important deliverable", "Answer-ready", "Frame as reproducible constrained-policy prototype"],
        [8, "Authentic recorded demonstration", "Action required", "Record a new end-to-end demonstration"],
        [9, "Original descriptive/experimental diagrams", "New original diagram supplied", "Audit remaining thesis figures"],
        [10, "Objective-aligned literature review", "Proposed alignment supplied; approved wording verification pending", "Reconcile with approved Chapter 1 objectives, then restructure Chapter Two"],
        [11, "Rationale under Table 3", "Correction wording ready", "Insert after verifying the exact Table 3 content"],
        [12, "Chapter-based numbering", "Correction rule ready", "Run whole-thesis caption/cross-reference audit"],
        [13, "Generated versus secondary data", "Evidence-ready", "Report mixed provenance explicitly"],
        [14, "Low performance values in abstract", "Corrected interpretation ready", "Retain real values; improve model only through new experiments"],
        [15, "Rationale for assumptions", "Assumption matrix ready", "Separate assumptions, delimitations and limitations"],
        [16, "Professional equations and numbering", "Equation set ready", "Insert with Word Equation Editor or LaTeX"],
        [17, "Algorithms instead of source code", "Draft pseudocode supplied; thesis replacement pending", "Remove code listings/screenshots from thesis"],
    ]

    variable_rows = [
        ["X1 / AI", "Observed adversarial intensity", "AI = attack epsilon / 0.30", "0 to 1", "0-0.167", ">0.167-0.333", ">0.333-1.000", "Reject outside domain"],
        ["X2 / PA", "Observed membership privacy pressure", "PA = 2 max(0, A_MIA - 0.50)", "0 to 0.50 fitted", "0-0.20", ">0.20-0.40", ">0.40-0.50", "Reject outside fitted domain"],
        ["X3 / DM", "Selected defense intensity", "Controller decision", "0 to 1", "Lowest-cost value with Y >= 0.70", "Lowest-cost value with Y >= 0.60", "DM = 1 maximum", "Reject/defer if target remains unmet"],
        ["X4 / GS", "Selected participation score", "GS = (K - 1) / 4", "0 to 1", "0.75-1.00 (4-5 clients)", "0.50 (3 clients)", "0-0.25 (1-2 clients)", "Controller requires K >= 3"],
        ["Y", "Predicted next-round efficiency", "Constrained linear response", "0 to 1", "Y >= 0.70", "0.60 <= Y < 0.70", "Y < 0.60", "Reject/defer"],
    ]

    coefficient_rows = []
    for name in ("B0", "B_AI", "B_PA", "B_DM", "B_GS"):
        row = coefficient_map[name]
        coefficient_rows.append([
            name,
            number(row["estimate"]),
            row["constraint"],
            "Satisfied" if str(row["constraint_satisfied"]).lower() == "true" else "Review",
            number(row["standard_error"]),
            f"[{number(row['ci_95_low'])}, {number(row['ci_95_high'])}]",
        ])

    federation_rows = [
        ["Logical clients", "5", "Fixed Phase-2 configuration", "One-process simulation"],
        ["Global rounds", "12", "Fixed Phase-2 configuration", "Recorded replay"],
        ["Local epochs", "2 per client per round", "Fixed Phase-2 configuration", "120 client-epochs in total"],
        ["Batch size", "64", "Fixed Phase-2 configuration", "Local mini-batch"],
        ["Training pairs", "640", "Fixed Phase-2 configuration", "128 paired rows per executed client"],
        ["Test pairs", "100", "Fixed Phase-2 configuration", "Paired batching rows"],
        ["Optimizer", "Adam", "Fixed", "Learning rate 0.001"],
        ["Aggregation", "Sample-weighted FedAvg", "Fixed", "Equal 0.20 weights in the recorded equal-size Phase-2 split"],
        ["DM", "0.00 to 1.00 in 0.05 steps", "Optimized in Phase 6", "21 candidates"],
        ["Participating clients / GS", "K in {3,4,5}; GS in {0.50,0.75,1.00}", "Optimized in Phase 6", "3 candidates"],
        ["Candidate policies", "21 x 3 = 63 per round", "Optimized in Phase 6", "Lowest-cost feasible candidate selected"],
        ["Targets", "Preferred Y >= 0.70; minimum Y >= 0.60", "Controller constraints", f"Recorded: {preferred_rounds} preferred, {managed_rounds} managed"],
    ]

    data_rows = [
        ["Face", "Secondary public data", "Labeled Faces in the Wild (LFW)", "3,023 images; 62 identities", "50 x 37; 1,850 features", "2,418 / 605", "Downloaded/cached by scikit-learn"],
        ["Fingerprint", "Generated locally", "Seeded procedural ridge-pattern generator", "800 images; 100 synthetic identities; 8 impressions each", "96 x 96; 9,216 features", "640 / 160", "No participant fingerprints collected"],
        ["Controller observations", "Generated locally", "Seeded balanced factorial design", "700 rows", "7 AI x 5 PA x 4 DM x 5 GS", "Not a biometric split", "Used only for Phase-6 coefficient recovery"],
    ]

    architecture_limitations = [
        ["Federation", "Synchronous central coordinator with five logical clients in one Python process", "No physical network, client devices or secure aggregation"],
        ["Client data", "Phase 1 saves identity-partitioned client files", "Phase 2 makes a fresh equal five-way in-memory split for paired batching"],
        ["Preprocessing", "Phase 1 standardizes each complete modality array before the 80/20 split", "Held-out distribution information enters scaler fitting, so results may be optimistic; a corrected rerun must fit each scaler on training data only and transform validation/test data"],
        ["Modalities", "Separate face and fingerprint branches/heads are trained", "Identities are independent and not co-enrolled"],
        ["Fusion", "A concatenated representation is calculated", "It is not connected to the final classification decisions; do not claim validated identity fusion"],
        ["Fingerprint evidence", "Synthetic fingerprint patterns support controlled code testing", "Recorded Phase-2 fingerprint accuracy is 0%; real-sensor generalization is unproven"],
        ["Privacy defense", "Gaussian gradient noise and membership-pressure evaluation are implemented", "The noise mechanism is DP-style; it has no formal privacy accountant or certified epsilon-delta guarantee"],
        ["Dynamic controller", "Phase 6 estimates coefficients and searches DM/GS on a synthetic design", "It does not yet retrain Phases 2-5 automatically in a closed loop"],
        ["Verification", "Operational FAR/FRR use claimed-identity face verification", "Those FAR/FRR values do not establish multimodal biometric verification"],
    ]

    literature_rows = [
        ["Objective 1: characterize attacks", "FGSM, PGD, poisoning and membership inference", "Goodfellow et al. (2015); Madry et al. (2018); Shokri et al. (2017)", "Defines evaluated threats and attack metrics"],
        ["Objective 2: quantify impact", "Accuracy, FAR, FRR, ASR and privacy leakage", "ISO/IEC 19795-1:2021; Shokri et al. (2017)", "Supports metric definitions, protocol disclosure and inference risk"],
        ["Objective 3: assess mechanisms", "FedAvg, adversarial training and privacy mechanisms", "McMahan et al. (2017); Madry et al. (2018); Dwork and Roth (2014)", "Explains model averaging, robustness and formal-DP boundary"],
        ["Objective 4: develop DO-FMBS", "Multimodal processing and constrained control", "Ross and Jain (2003); Kraft (1988)", "Supports complementary modalities and SLSQP-based constrained estimation/control"],
        ["Objective 5: validate DO-FMBS", "Clean, attack, privacy, statistics and runtime evidence", "ISO/IEC 19795-1:2021", "Requires test protocol, denominators and applicability limits"],
        ["Data selection", "Unconstrained public faces and synthetic test fingerprints", "Huang et al. (2007); NIST IR 8361 (2021)", "Supports LFW provenance and limits claims from synthetic biometric data"],
    ]

    assumption_rows = [
        ["Identical model architecture at every client", "FedAvg averages corresponding parameter arrays", "Direct averaging is invalid when shapes differ", "Assumption"],
        ["Coordinator follows the declared protocol", "The simulation threat model focuses on client/model attacks", "A malicious server needs additional controls", "Assumption"],
        ["AI and PA are available before the control decision", "The controller needs current pressure inputs", "DM and GS cannot adapt without observable inputs", "Assumption"],
        ["Inputs remain inside fitted Phase-6 domains", "The response equation is supported only within its design", "Prediction is rejected rather than extrapolated", "Assumption"],
        ["Five clients and selected attack strengths", "Makes the experiment finite and reproducible", "Does not generalize automatically to other scales", "Delimitation"],
        ["One-process logical federation", "Permits controlled research execution on available hardware", "Does not measure real network faults or latency", "Limitation"],
        ["Synthetic fingerprints", "Avoids collecting participant fingerprint data and permits seeded tests", "Does not establish sensor/population validity", "Limitation/design choice"],
        ["Independent face and fingerprint identities", "Public face data and generated fingerprints have different label spaces", "Same-person multimodal verification cannot be claimed", "Limitation"],
        ["Full-array scaling before the train/test split", "This is how the current Phase-1 script prepares both modality arrays", "Held-out distribution information enters preprocessing and may make reported performance optimistic; refit scalers on training data only in the corrected rerun", "Limitation"],
    ]

    correction_rows = [
        [1, "Chapter 4 variable section", "Insert definitions, values, bands, domain guard and critical example", "Ready for insertion", "Phase-6 fit/stress artifacts"],
        [2, "Chapter 4 mathematical formulation", "Replace inconsistent labels with explicit X-to-named mapping", "Ready for insertion", "Coefficient CSV/model_fit.json"],
        [3, "Chapter 4 optimization", "Explain two-stage fitting and control search", "Ready for insertion", "Model fit and schedule"],
        [4, "Chapter 4 FL configuration", "Separate fixed training settings from optimized DM/GS", "Ready for insertion", "Phase-2 metrics/Phase-6 settings"],
        [5, "Chapter 4 architecture/method", "Add source, selection rationale and simulation boundaries", "Ready for insertion", "Phase scripts and artifacts"],
        [6, "Chapters 1, 2 and 4", "Strengthen modality rationale; state unproven fusion limitation", "Ready; extension is future work", "Source/model audit"],
        [7, "Abstract/conclusion", "State the reproducible constrained-policy prototype as the deliverable", "Ready for insertion", "Six-phase replay"],
        [8, "Presentation appendix", "Record authentic repository-to-results demonstration", "New recording required", "Logs exist; video must be newly recorded"],
        [9, "Chapter 4 figures", "Use project-specific experimental/control diagram", "New diagram supplied; thesis audit required", "Original generated figure"],
        [10, "Chapter 2", "Verify the approved Chapter 1 objective wording, reconcile the proposed matrix, then close each section with a research gap", "Approved-objective verification and manuscript work required", "Proposed objective-literature matrix"],
        [11, "Immediately below exact Table 3", "Add value provenance, rationale, unit/range, limitation and citation", "Placement requires Table 3 verification", "Table content not reproduced here"],
        [12, "Whole thesis", "Apply chapter-based automatic captions and cross-references", "Whole-thesis audit required", "Word caption/cross-reference functions"],
        [13, "Data-source sections and abstract", "Declare mixed provenance", "Ready for insertion", "Phase-1 arrays/generator"],
        [14, "Abstract", "Use synchronized metrics with scope and denominator", "Corrected wording ready", "Phase-5/evaluator artifacts"],
        [15, "Chapter 1", "Add rationale and violation consequence; reclassify limitations/delimitations", "Ready for insertion", "Assumption matrix"],
        [16, "Whole thesis", "Replace screenshots/plain text with numbered editable equations", "Equation set ready", "Equations R.1-R.11"],
        [17, "Chapter 4/appendices", "Replace full source listings with six algorithms; retain repository link", "Manuscript work required", "Function traceability table"],
    ]

    trace_rows = [
        ["LFW acquisition", "phase1_data_preparation.py", "fetch_lfw_people", "do_fmbs_data/face_data.npz"],
        ["Fingerprint generation", "phase1_data_preparation.py", "generate_one_fingerprint", "do_fmbs_data/fingerprint_data.npz"],
        ["Federated local update", "phase2_federated_learning.py", "EnhancedFedAvg.client_update", "baseline_metrics.pkl"],
        ["FedAvg aggregation", "phase2_federated_learning.py", "EnhancedFedAvg.aggregate", "baseline_model.keras"],
        ["FGSM/PGD and ASR", "phase3_adversarial_attacks.py", "evaluate_attack", "attack_results.pkl"],
        ["DP-style noise", "phase4_defense_mechanisms.py", "train_with_dp", "final_results.pkl"],
        ["Adversarial training", "phase4_defense_mechanisms.py", "train_with_art", "do_fmbs_model.keras"],
        ["Robustness/statistics", "phase5_model_validation.py", "evaluate_model_robustness", "validation_results.pkl"],
        ["Coefficient estimation", "scripts/constrained_dynamic_model.py", "fit_equation", "model_fit.json; coefficient_estimates.csv"],
        ["DM/GS policy search", "scripts/constrained_dynamic_model.py", "optimize_controls", "dynamic_optimization_schedule.csv"],
        ["Operational FAR/FRR/PA", "scripts/measure_operational_biometric_metrics.py", "claimed-identity evaluator", "clean_far_frr_acceptance.csv; privacy_pa_evaluation.csv"],
        ["Runtime", "scripts/run_isolated_instrumented_pipeline.py", "timed phase execution", "phase_runtime_registry.csv"],
    ]

    result_rows = [
        ["Phase-2 baseline face classification", percent(evidence.baseline["face_accuracy"]), "100 paired test rows"],
        ["Phase-2 baseline fingerprint classification", percent(evidence.baseline["finger_accuracy"]), "100 paired test rows"],
        ["Phase-4 combined clean face accuracy", percent(evidence.defenses["comparison"]["do_fmbs"]["face_accuracy"]), "605 face samples"],
        ["Phase-4 combined conditional FGSM ASR", percent(evidence.defenses["comparison"]["do_fmbs"]["attack_success_rate"]), "epsilon 0.05; clean-correct denominator"],
        ["Phase-5 final face classification accuracy", percent(evidence.validation["classification_metrics"]["do_fmbs"]["accuracy"]), "First 100 validation faces"],
        ["Operational final FAR", percent(final["FAR"], 3), f"{final['FAR_numerator']}/{final['FAR_denominator']} impostor claims"],
        ["Operational final FRR", percent(final["FRR"], 3), f"{final['FRR_numerator']}/{final['FRR_denominator']} genuine claims"],
        ["Operational PA", number(recorded.pa_row["PA"], 3), f"Membership accuracy {percent(recorded.pa_row['membership_accuracy'])}"],
        ["Phase-6 response fit", f"R-squared {number(fit['r_squared'])}; RMSE {number(fit['rmse'])}", "700 seeded synthetic conditions"],
        ["Active Phase 1-6 runtime", f"{total_runtime:.6f} seconds", "25 minutes 22.308 seconds"],
    ]

    parts = [
        "# FORMAL RESPONSE TO MOCK-PRESENTATION COMMENTS",
        "",
        "## A Privacy-Preserving Dynamic Optimization Model for Mitigating Adversarial Attacks in Federated Multimodal Biometric Systems (DO-FMBS)",
        "",
        "**Candidate:** Lydiah Moraa Machora (I152/P/9515/23)",
        "",
        "**Supervisors:** Dr. Richard Omollo and Dr. Gonza Omoro",
        "",
        "**Stage:** Mock-presentation correction and resubmission",
        "",
        f"**Document generated:** {generated_at}",
        "",
        "---",
        "",
        "## Document purpose",
        "",
        "This standalone document gives a direct, professional and evidence-linked response to each of the 17 mock-presentation comments. It is a response register and oral-defense aid; it does not replace the thesis. Each response uses the same five blocks: the examiner's concern, the direct answer, the supporting evidence, the manuscript correction, and a concise answer to present orally.",
        "",
        "## Evidence declaration",
        "",
        f"All numerical results in this response were loaded from the synchronized replay `{recorded.run_dir.name}` and the operational evaluator `{recorded.evaluator_dir.name}`. The six phase commands completed successfully. Active command time was {total_runtime:.6f} seconds. Phase-6 coefficients and Y values come from a seeded 700-condition synthetic controller experiment; they are not neural-network weights and are not estimates from 700 federated-training runs. No performance value has been raised, suppressed or relabelled to make the result appear stronger.",
        "",
        "The status descriptions distinguish evidence already available from corrections that still require insertion into the thesis, a whole-manuscript audit, an authentic video recording or a future experiment. This prevents a proposed correction from being presented as completed evidence.",
        "",
        "## Executive status summary",
        "",
        md_table(["No.", "Examiner issue", "Current disposition", "Required thesis/presentation action"], status_rows),
        "",
        "## Key synchronized results",
        "",
        md_table(["Evidence scope", "Recorded value", "Denominator or interpretation"], result_rows),
        "",
    ]

    parts.extend(response_blocks(
        1,
        "Variable values, ideal operating bands and out-of-range deployment",
        "Did the simulation-constrained model attach values to its variables, define ideal ranges, and specify what happens when deployed values fall outside those ranges?",
        [
            "Yes. AI and PA are observed pressure variables; DM and GS are controller decisions; Y is the predicted next-round efficiency response. Lower AI and PA are desirable. DM and GS do not have one universal ideal value: the controller chooses the lowest-cost values that satisfy the response target for the current round.",
            "The preferred decision satisfies Y >= 0.70. A managed decision satisfies 0.60 <= Y < 0.70. If maximum admissible controls cannot attain Y >= 0.60, deployment is rejected or deferred. Values outside the fitted variable domains are rejected instead of extrapolated.",
        ],
        [
            md_table(["Variable", "Definition", "Domain", "Preferred", "Managed", "Critical", "Action"], [row[:1] + [row[1] + "; " + row[2]] + row[3:] for row in variable_rows]),
            "At the modeled critical condition AI=0.666667, PA=0.50, DM=1 and GS=1, the fitted equation gives Y=0.586453. Because this remains below 0.60 even at maximum controls, the recorded decision is reject or defer. An out-of-domain example with attack epsilon=0.35 gives AI=1.166667 and PA=0.60; the domain guard rejects it before prediction.",
            "The bands are researcher-defined decision rules for this simulation. They must not be described as international biometric deployment standards.",
        ],
        [
            "Insert the complete range table, the preferred/managed/critical interpretation, and both rejection cases into Sections 4.4.3 and 4.4.6. Add the qualification that the cutoffs are simulation governance choices requiring external calibration before real deployment.",
        ],
        "Yes. The variables have explicit values and fitted domains. I accept a preferred policy at Y of at least 0.70, a managed policy at Y of at least 0.60, and I reject or defer a round when the minimum target is infeasible or the inputs are outside the fitted domain.",
    ))

    parts.extend(response_blocks(
        2,
        "Alignment of definitions, equations and computation ranges",
        "The variable definitions must align with the proposed model; the equations and value ranges must be reviewed for consistent computation.",
        [
            "The earlier X1-X4 notation and the named AI-PA-DM-GS notation describe the same four-predictor response model. The corrected manuscript must show the mapping explicitly and use one residual symbol consistently.",
        ],
        [
            "The general equation is:",
            *equation(r"Y_i=B_0+B_1X_{1i}+B_2X_{2i}+B_3X_{3i}+B_4X_{4i}+u_i\quad\mathrm{(R.1)}"),
            "The mapping is X1=AI, X2=PA, X3=DM and X4=GS. Substitution gives:",
            *equation(r"Y_i=B_0+B_{AI}AI_i+B_{PA}PA_i+B_{DM}DM_i+B_{GS}GS_i+u_i\quad\mathrm{(R.2)}"),
            "Because round-t measurements and controls predict the round-(t+1) response, the implemented predictor is:",
            *equation(r"\widehat{Y}_{t+1}=B_0+B_{AI}AI_t+B_{PA}PA_t+B_{DM}DM_t+B_{GS}GS_t\quad\mathrm{(R.3)}"),
            "AI and PA are parallel observed predictors; the model does not claim that AI mathematically causes PA. DM and GS are decisions. Y is a synthetic efficiency response, not accuracy, FAR, FRR or ASR.",
        ],
        [
            "Use Equations (R.1)-(R.3) in the mathematical-formulation section, define every symbol immediately below the first equation, and keep X1-X4 visible as an explicit equivalence table. Remove any wording that describes AI -> PA -> DM -> GS -> Y as a causal chain.",
        ],
        "Both forms are correct, but the mapping must be explicit: X1 is AI, X2 is PA, X3 is DM and X4 is GS. They enter one multiple-response equation; they are not a causal chain.",
    ))

    parts.extend(response_blocks(
        3,
        "What is dynamic and how optimization occurs",
        "The model must identify exactly what changes dynamically and explain the optimization mechanism.",
        [
            "The model is dynamic in its decision logic because it accepts a round-specific AI_t/PA_t state, after which it reselects DM_t and GS_t and predicts Y_(t+1). In the recorded demonstration the 12 AI/PA states are seeded, fixed inputs in Phase 6 rather than live observations from the neural phases. There are two distinct optimization operations: constrained coefficient estimation and round-specific policy selection.",
        ],
        [
            "First, SLSQP minimizes squared response error subject to B_AI<=0, B_PA<=0, B_DM>=0 and B_GS>=0. The fit recorded R-squared 0.968064, cross-validation R-squared 0.967254, RMSE 0.024528 and maximum constraint violation 0. These statistics validate recovery of the seeded synthetic response relation; they do not validate biometric recognition accuracy.",
            "Second, the controller evaluates 21 DM values and three permitted client counts, producing 63 candidate policies in each round. Its objective is:",
            *equation(r"J_t=0.60DM_t^2+0.40GS_t^2+0.15(DM_t-DM_{t-1})^2+0.10(GS_t-GS_{t-1})^2\quad\mathrm{(R.4)}"),
            "The first two terms penalize defense and participation cost; the last two discourage abrupt policy changes. It first searches for Y>=0.70, then Y>=0.60, and otherwise rejects or defers.",
            f"The recorded 12-round schedule produced {preferred_rounds} preferred decisions and {managed_rounds} managed decisions.",
        ],
        [
            "Add a two-stage algorithm: Algorithm A for constrained coefficient estimation and Algorithm B for per-round DM/GS search. State explicitly that Phase 6 is presently an independently executed controller demonstration and is not yet a closed-loop retraining trigger for Phases 2-5.",
            f"[[FIGURE:{FIGURE_NAME}|Figure 1: Evidence boundaries and dynamic DO-FMBS control process (Researcher, 2026). The executed Phase 1-5 biometric experiment is separated from the Phase-6 synthetic controller demonstration. Phase 5 records classification, robustness and statistical evidence. Post-run evaluation adds FAR, FRR and PA; the runtime registry adds phase durations; Chapter 4 separately reports an analytical raw-weight communication equivalent.]]",
        ],
        "Dynamic means that the controller treats specified AI and PA inputs as the observations for each round, then re-optimizes defense intensity and client participation over 63 feasible candidates. The recorded 12-round AI/PA path is seeded and fixed in the controller script; it is not live telemetry from Phases 3 or 4. Coefficient fitting and round control are separate operations.",
    ))

    parts.extend(response_blocks(
        4,
        "Optimization values within federated learning",
        "The values used for optimization and federation must be stated precisely.",
        [
            "The recorded federated-training settings are fixed experimental hyperparameters. The Phase-6 adaptive search optimizes only DM and active-client participation represented by GS; it does not claim to optimize learning rate, local epochs, batch size or number of global rounds.",
        ],
        [
            md_table(["Setting", "Value/search space", "Role", "Interpretation"], federation_rows),
            "The sample-weighted aggregation rule is:",
            *equation(r"w^{t+1}=\sum_{k=1}^{K}\frac{n_k}{\sum_{j=1}^{K}n_j}w_k^{t+1}\quad\mathrm{(R.5)}"),
            "The recorded Phase-2 execution used 128 paired rows at each of five logical clients, so each update had weight 128/640=0.20. Twelve rounds, five clients and two local epochs produced 60 local updates and 120 client-epochs.",
        ],
        [
            "Replace any statement that says all federation parameters are dynamically optimized. Present a fixed-versus-optimized configuration table. A future hyperparameter study would be required to optimize rounds, epochs, batch size and learning rate.",
        ],
        "The recorded federation used five clients, 12 rounds, two local epochs, batch size 64 and Adam at learning rate 0.001. Phase 6 optimizes only defense intensity and whether three, four or five clients participate.",
    ))

    parts.extend(response_blocks(
        5,
        "Deployed architecture, data source and scientific mechanism",
        "The study must clearly describe the federated architecture, data source, selection rationale and scientific mechanism.",
        [
            "The implemented architecture is a synchronous, centrally coordinated, five-client FedAvg simulation. A server distributes a global model, each logical client performs local training, and the server combines returned parameter arrays using sample-weighted averaging. The experiment runs in one Python process; it is not a physically distributed deployment.",
            "LFW was selected as a reproducible public face dataset containing uncontrolled pose, illumination and appearance variation. The installed `fetch_lfw_people` loader acquires the funneled archive from https://ndownloader.figshare.com/files/5976015; its official documentation is https://scikit-learn.org/stable/modules/generated/sklearn.datasets.fetch_lfw_people.html. Fingerprints were generated locally to permit seeded, privacy-conscious pipeline testing without collecting sensitive participant fingerprints; therefore, the fingerprint data has no download URL.",
        ],
        [
            md_table(["Component", "Implemented mechanism", "Scientific boundary"], architecture_limitations),
            "The experimental sequence is data preparation, federated baseline training, evasion/inference/poisoning attacks, defense-model comparison, validation, and a separate constrained-controller demonstration. McMahan et al. (2017) supports iterative federated averaging; Huang et al. (2007) supports the LFW data source; attack and defense choices are aligned with the cited primary literature in the objective-literature matrix.",
            "A preprocessing limitation is also disclosed: Phase 1 calls StandardScaler.fit_transform on each complete modality array before creating the 80/20 split. This lets held-out distribution information influence feature scaling and may make downstream results optimistic. A corrected rerun must fit each scaler on the training partition only, then use that fitted scaler to transform validation and test data.",
        ],
        [
            "Revise Sections 4.1-4.3 and 4.5 to name the coordinator, logical clients, exchanged parameter arrays, aggregation equation, sources, selection reasons and all architecture boundaries. Disclose the current full-array scaler fitting and schedule a corrected train-only-scaler rerun. Do not use the word deployed to imply a production network; use implemented simulation architecture unless actual distributed deployment evidence is added.",
        ],
        "I implemented a synchronous five-client FedAvg simulation. LFW supplies reproducible unconstrained faces, fingerprints are generated locally for controlled testing, and the six phases provide a traceable attack-defense-validation experiment. It is a one-process research simulation, not a production network.",
    ))

    parts.extend(response_blocks(
        6,
        "Rationale for face and fingerprint and extension to other modalities",
        "The thesis must justify the two selected modalities and explain whether the model can accept another modality.",
        [
            "Face and fingerprint were selected because they represent structurally different biometric evidence: face supports convenient non-contact acquisition, while fingerprint ridge structure provides a different biometric characteristic. Multibiometric literature motivates complementary evidence, but this implementation does not yet prove a fusion advantage.",
            "Another modality such as iris, voice or palmprint can be added architecturally, but it is not a plug-in change. It requires a modality-specific loader, preprocessing and encoder, co-enrolled identity labels, a defined fusion rule, loss and batching changes, attacks/defenses suited to the modality, retraining, threshold calibration and complete validation.",
        ],
        [
            "The recorded data has independent face and fingerprint identity spaces, the final decisions are produced by separate heads, and the calculated concatenated tensor is not connected to those decisions. Phase-2 fingerprint accuracy is 0%. Consequently, the current evidence supports dual-branch multimodal processing, not same-person multimodal biometric verification or a measured fusion benefit.",
            "Ross and Jain (2003) explain why multiple biometric evidences can mitigate limitations of a single trait, while also emphasizing the need for an effective fusion scheme. That supports the design rationale, not a claim that the current fusion is already validated.",
        ],
        [
            "Strengthen the justification in Chapters 1 and 2, then state the implementation boundary in Chapter 4 and the limitation in Chapter 5. Describe co-enrolled real fingerprint/face evaluation and learned score-level or feature-level fusion as future work.",
        ],
        "Face and fingerprint are complementary biometric types, but my current data is not co-enrolled and the fingerprint head did not validate successfully. A third modality is possible only after adding its encoder, co-enrolled identities, a fusion mechanism and a new validation study.",
    ))

    parts.extend(response_blocks(
        7,
        "Most important research deliverable",
        "The candidate must identify the single most important output of the research.",
        [
            "The principal deliverable is an auditable Phase 1-5 attack/defense experiment plus a separate reproducible constrained-policy demonstrator that maps specified AI/PA inputs to explicit choices of defense intensity and client participation. Together they preserve a complete evidence trail, but the current code does not pass measured Phase-3 or Phase-4 values into Phase 6.",
        ],
        [
            "The deliverable includes executable Phase 1-6 modules, synchronized logs and artifacts, constrained coefficients with all sign restrictions satisfied, a 12-round controller schedule, an operational face-verification evaluation and source-to-artifact traceability. Its strongest contribution is the auditable policy mechanism, not a claim of deployment readiness.",
            f"Recorded supporting values include final face classification accuracy {percent(evidence.validation['classification_metrics']['do_fmbs']['accuracy'])}, FAR {percent(final['FAR'],3)}, FRR {percent(final['FRR'],3)}, PA {number(recorded.pa_row['PA'],2)}, and {total_runtime:.6f} seconds active Phase 1-6 runtime.",
        ],
        [
            "Use this deliverable statement consistently in the abstract, Chapter 1 contribution section, Chapter 4 summary and Chapter 5 conclusion. Call it a research prototype and policy demonstrator, not a finished security product.",
        ],
        "My most important deliverable is the auditable Phase 1-5 attack/defense experiment together with a separate constrained-policy demonstrator that maps specified AI and PA inputs to DM and GS choices. Wiring recorded attack/privacy metrics into closed-loop retraining is future work.",
    ))

    parts.extend(response_blocks(
        8,
        "Authentic recorded demonstration of the pipeline",
        "An authentic recorded demonstration or video should describe the simulation pipeline.",
        [
            "This recommendation is accepted. The existing logs and screenshots support traceability, but an authentic narrated video must be newly recorded. The report does not represent an unrecorded video as completed evidence.",
        ],
        [
            "The video should show: repository path and current commit; Python/library versions; creation of a new isolated run directory; exact commands; Phase-1 data acquisition/generation; Phase-2 federated rounds; Phase-3 attacks; Phase-4 defenses; Phase-5 validation; Phase-6 control calculation; saved results, runtime registry and hashes. If long training is time-compressed, the edit must be disclosed and the timestamps/logs kept visible.",
            "Recommended reproducible command: `.\\venv\\Scripts\\python.exe scripts\\run_isolated_instrumented_pipeline.py --run-dir runtime_recording\\mock_resubmission_demo`.",
        ],
        [
            "Add the completed recording as a presentation appendix item, identify its date and commit hash, and retain the unedited terminal logs. Record a new run rather than narrating old screenshots as though they were live execution.",
        ],
        "I have reproducible logs and artifacts, and I accept that the resubmission also needs an authentic narrated run. I will record a new isolated execution showing the commands, outputs, runtimes and saved evidence from start to finish.",
    ))

    parts.extend(response_blocks(
        9,
        "Original descriptive and experimental diagrams",
        "Copied diagrams should be avoided; the thesis should provide clear project-specific descriptive and experimental diagrams.",
        [
            "The correction uses original project diagrams generated from the implemented process. The diagrams distinguish executed data/model flow from the separate Phase-6 controller experiment and avoid suggesting connections that the code does not implement.",
        ],
        [
            "Figure 1 in this response shows the evidence boundaries and the controller decision cycle. A complete thesis figure set should include: the Phase 1-6 architecture; baseline-attacks-defenses-validation experimental design; and the per-round observe-search-evaluate-accept/defer controller flow.",
            "Every arrow must correspond either to an implemented artifact flow or to a clearly labelled design/future integration. Captions should identify the researcher and year, and externally adapted visual elements must be cited.",
        ],
        [
            "Replace generic figures in Chapter 4 with project-specific versions and audit all remaining thesis diagrams for originality, source attribution and consistency with the implementation. Use the diagram in this response as a correction asset, not as proof of an unimplemented closed loop.",
        ],
        "I have prepared an original project-specific diagram that clearly separates the executed biometric phases from the synthetic controller. I will replace generic thesis artwork and audit the remaining figures so that every relationship is implemented or explicitly labelled as a design linkage.",
    ))

    parts.extend(response_blocks(
        10,
        "Literature review aligned with the study objectives",
        "The literature review should be organized according to the research objectives.",
        [
            "Chapter Two should be reorganized by objective rather than as a sequence of author summaries. Each objective-aligned section should define the problem, synthesize findings, identify limitations, state the research gap and explain how that gap informed DO-FMBS.",
            "The exact approved Chapter 1 objective wording is not present in the inspected project evidence. Therefore, the five-row structure below is a proposed thematic alignment, not a verified quotation or final numbering of the approved objectives. It must be reconciled row by row with the supervisor-approved Chapter 1 objectives before insertion into Chapter Two.",
        ],
        [
            "**Proposed objective-to-literature alignment - confirm against the approved Chapter 1 objectives before insertion:**",
            md_table(["Research objective", "Required literature theme", "Core authoritative basis", "Connection to this study"], literature_rows),
            "This matrix groups the available research themes into a defensible five-part sequence; it does not establish that these labels reproduce the approved Chapter 1 objectives verbatim. The final headings, order and wording must follow the approved objectives.",
            "After that reconciliation, the review should use primary papers and authoritative standards wherever possible. It must distinguish formal differential privacy from the project's DP-style noise, and distinguish multibiometric fusion theory from the current dual-head implementation boundary.",
        ],
        [
            "First obtain or verify the exact supervisor-approved Chapter 1 objectives and map each approved objective to the themes and sources in the proposed matrix. Then rebuild Chapter Two using the reconciled headings. End each main subsection with four labelled statements: synthesis, limitation, gap and DO-FMBS response. Retain a conventional conceptual framework only after the objective-specific synthesis is complete.",
        ],
        "I have prepared a proposed objective-to-literature alignment. Before inserting it, I will reconcile every row with the exact approved Chapter 1 objective wording; I will then organize Chapter Two by those approved objectives and end each section with the limitation, research gap and study response.",
    ))

    parts.extend(response_blocks(
        11,
        "Rationale and citation below Table 3",
        "The information under Table 3 requires a clear rationale or relevant citation.",
        [
            "The correction must identify the provenance and purpose of every value in the actual Table 3. A numerical value may be literature-derived, experimentally measured, researcher-defined or fitted; those categories must not be mixed.",
        ],
        [
            "For AI/PA/DM/GS bands, the appropriate statement is that the transformations and decision bands are researcher-defined simulation governance values within the fitted domain. Literature supports the attack, privacy and federation concepts, but the numerical cutoffs are not claimed to come from ISO or another external deployment standard.",
            "For experimental hyperparameters, the table must report the recorded command/configuration and explain why the value was held fixed. For observed metrics, it must report the artifact and denominator. For standard definitions such as biometric error rates, it should cite ISO/IEC 19795-1:2021.",
        ],
        [
            "Insert a paragraph immediately below the exact Table 3 after verifying its final contents. For each row, provide provenance, unit/domain, rationale, objective served, limitation and source. Because the exact thesis Table 3 is not reproduced in this response package, its final row-by-row citation cannot responsibly be guessed here.",
        ],
        "I will identify each Table 3 value as literature-derived, measured, fitted or researcher-defined. My AI/PA/DM/GS cutoffs are simulation decision bands, so I will justify them without presenting them as external standards.",
    ))

    parts.extend(response_blocks(
        12,
        "Chapter-based naming of tables and figures",
        "Tables and figures must be numbered according to the chapters in which they appear.",
        [
            "The correction rule is chapter-based sequential numbering: Table 4.1, Table 4.2, Figure 4.1, Figure 4.2 and Equation 4.1 for Chapter Four. Captions and in-text citations must use automatic cross-references.",
        ],
        [
            "Table captions belong above tables; figure captions belong below figures; equations are numbered consistently at the right. The Word lists of tables and figures must be regenerated after all captions and cross-references are updated.",
        ],
        [
            "Run a whole-thesis audit, replace labels such as `Table 3` with chapter identifiers such as `Table 3.1`, remove duplicate numbers, update every in-text reference and regenerate the automatic lists. Chapter 4 alone cannot establish whole-thesis compliance.",
        ],
        "I will apply chapter-based numbering throughout and audit the complete thesis so that every table, figure and equation has an automatic chapter number and a valid cross-reference.",
    ))

    parts.extend(response_blocks(
        13,
        "Generated data versus secondary data",
        "The thesis must state whether its data was generated or obtained from a secondary source.",
        [
            "The study uses mixed provenance: face images are secondary public LFW data; fingerprints are generated locally; and the Phase-6 response observations are a seeded synthetic factorial design.",
        ],
        [
            md_table(["Data", "Provenance", "Source/generator", "Quantity", "Representation", "Split/design", "Qualification"], data_rows),
            "The installed LFW fetcher acquires its funneled archive from https://ndownloader.figshare.com/files/5976015 and stores raw material in the scikit-learn cache; the official loader documentation is https://scikit-learn.org/stable/modules/generated/sklearn.datasets.fetch_lfw_people.html. The project stores normalized arrays in `do_fmbs_data/face_data.npz`. The fingerprint generator builds ridge patterns, adds seeded local variation, normalizes them and stores arrays in `do_fmbs_data/fingerprint_data.npz`; individual participant fingerprints were neither downloaded nor collected, and there is no fingerprint download URL. The LFW technical-report URL in the references documents the dataset literature and is distinct from the acquisition URL.",
        ],
        [
            "State mixed provenance consistently in the abstract, data-source section, ethics/data statement and limitations. Cite Huang et al. (2007) for LFW. Describe the fingerprint and Phase-6 records as synthetic, and do not label them as real secondary biometric samples.",
        ],
        "The face images are secondary LFW data. The 800 fingerprints and 700 controller observations are generated locally. No participant fingerprints were collected.",
    ))

    parts.extend(response_blocks(
        14,
        "Interpretation of low performance values in the abstract",
        "The performance values reported in the abstract appear too low and require correction or explanation.",
        [
            "The scientific correction is to report the synchronized measurements with scope and denominators, not to increase or selectively replace them. The results demonstrate improvement in the face branch but also reveal unresolved fingerprint and high-intensity robustness limitations.",
        ],
        [
            f"On the same Phase-5 classification evaluation, face accuracy increased from {percent(evidence.validation['classification_metrics']['baseline']['accuracy'])} to {percent(evidence.validation['classification_metrics']['do_fmbs']['accuracy'])}, an absolute increase of 39 percentage points. Operational face verification recorded FAR {percent(final['FAR'],3)} ({final['FAR_numerator']}/{final['FAR_denominator']}) and FRR {percent(final['FRR'],3)} ({final['FRR_numerator']}/{final['FRR_denominator']}). PA was {number(recorded.pa_row['PA'],2)}. At attack epsilon 0.10, the Phase-5 final classification ASR was {percent(evidence.validation['robustness']['do_fmbs'][2]['attack_success_rate'])}. At epsilon 0.20, recorded adversarial accuracy was only {percent(evidence.validation['robustness']['do_fmbs'][4]['accuracy'])} for DO-FMBS, compared with {percent(evidence.validation['robustness']['baseline'][4]['accuracy'])} for the baseline, confirming that high-intensity robustness remains unresolved.",
            "The Phase-2 fingerprint-head accuracy was 0.00%, and Phase-6 R-squared 0.968064 belongs to synthetic coefficient recovery, not biometric performance. Both qualifications must remain visible.",
            "Recommended abstract sentence: `The final face classifier achieved 65.00% accuracy, compared with 26.00% for the Phase-5 baseline. Under clean claimed-identity face verification, FAR was 7.852% (1,504/19,154 impostor claims), FRR was 11.146% (35/314 genuine claims), and membership privacy pressure was 0.34. At attack epsilon 0.10, classification ASR reached 70.09%, indicating that stronger high-intensity robustness remains necessary.`",
        ],
        [
            "Replace vague performance claims in the abstract with the proposed synchronized sentence. Do not describe 65% as deployment-grade without an external benchmark. A stronger result can only come from a documented new model/data experiment and a full rerun.",
        ],
        "I have not inflated the values. I now report the experiment, denominator and limitation: face accuracy improved from 26% to 65%, clean face-verification FAR was 7.852% and FRR 11.146%, but fingerprint recognition and strong-attack robustness still require further research.",
    ))

    parts.extend(response_blocks(
        15,
        "Rationale for the study assumptions in Chapter One",
        "Each study assumption requires a rationale, and assumptions should be separated from delimitations and limitations.",
        [
            "The assumptions define conditions required for the implemented methods to remain valid. The corrected Chapter One should explain why each condition is necessary and what conclusion fails when it is violated.",
        ],
        [
            md_table(["Statement", "Rationale", "If violated", "Correct classification"], assumption_rows),
            "The distinction is important: an assumption is accepted for the analysis; a delimitation is a deliberate scope choice; and a limitation is a constraint that restricts interpretation or generalization.",
        ],
        [
            "Replace the current assumption list with the matrix above. Move the five-client scope and selected attack values to delimitations. Move one-process execution, synthetic fingerprints, independent identities, full-array preprocessing before the split, absence of secure aggregation and absence of real network behavior to limitations. State that the corrected experiment will fit scalers on training data only.",
        ],
        "My assumptions are the conditions the algorithms require, such as compatible client models and observable AI/PA. Five clients is a delimitation, while synthetic fingerprints and one-process execution are limitations; I will separate these categories in Chapter One.",
    ))

    parts.extend(response_blocks(
        16,
        "Professional equation tools and appropriate numbering",
        "The equations should be written with professional tools and numbered consistently.",
        [
            "All thesis equations should be created with Word Equation Editor, MathType or LaTeX, remain editable, and use chapter-based numbering. Equation screenshots and manually typed pseudo-equations should be removed.",
        ],
        [
            "The fitted response equation is:",
            *equation(r"\widehat{Y}_{t+1}=0.400738-0.182145AI_t-0.227329PA_t+0.261393DM_t+0.159417GS_t\quad\mathrm{(R.6)}"),
            "The transformations are:",
            *equation(r"AI_t=\frac{\epsilon_{attack,t}}{0.30}\quad\mathrm{(R.7)}"),
            *equation(r"PA_t=2\max(0,A_{MIA,t}-0.50)\quad\mathrm{(R.8)}"),
            *equation(r"GS_t=\frac{K_t-1}{4}\quad\mathrm{(R.9)}"),
            "The response and search constraints are:",
            *equation(r"\widehat{Y}_{t+1}\geq0.60,\quad0\leq DM_t\leq1,\quad GS_t\in\{0.50,0.75,1.00\}\quad\mathrm{(R.10)}"),
            "The coefficient sign constraints are:",
            *equation(r"B_{AI}\leq0,\quad B_{PA}\leq0,\quad B_{DM}\geq0,\quad B_{GS}\geq0\quad\mathrm{(R.11)}"),
            md_table(["Coefficient", "Estimate", "Restriction", "Verification", "Standard error", "95% confidence interval"], coefficient_rows),
        ],
        [
            "Insert the equations with the thesis's Chapter 4 numbering, align equation numbers on the right, define every symbol once, and cross-reference equations automatically. Use u_i for the regression residual so it is not confused with attack epsilon.",
        ],
        "I have prepared a consistent equation set and will insert it with Word Equation Editor or LaTeX using Chapter 4 numbering. The X1-X4 mapping, fitted coefficients, transformations, objective and constraints use one notation.",
    ))

    parts.extend(response_blocks(
        17,
        "Pseudocode and algorithms instead of source-code listings",
        "Source code should not appear in the thesis; the simulation modules should be presented as pseudocode or algorithms.",
        [
            "The recommendation is accepted. The thesis should contain concise, language-independent algorithms showing inputs, processing, decisions and outputs. The full executable implementation remains in the repository for reproducibility.",
        ],
        [
            "The six required algorithms are: Algorithm 4.1, data preparation and client allocation; Algorithm 4.2, local training and sample-weighted FedAvg; Algorithm 4.3, FGSM/PGD, membership and poisoning evaluation; Algorithm 4.4, defense-model training; Algorithm 4.5, validation and operational metrics; and Algorithm 4.6, constrained DM/GS selection.",
            "Each algorithm should name its inputs, outputs, loop conditions, random seed, metric denominator and saved artifact. The source-to-artifact table in Appendix H preserves implementation traceability without reproducing code in the thesis.",
            "Draft thesis-ready pseudocode is supplied in Appendix I of this response. It is language-independent and separates Phase-5 validation from the post-run FAR/FRR/PA and runtime evaluation.",
        ],
        [
            "Remove source-code screenshots and long code listings from Chapter 4 and the submitted appendix. Replace them with the six algorithms. Keep the repository URL, version/commit information and reproducibility commands in a software-availability statement.",
        ],
        "I will present the six modules as algorithms with inputs, operations and outputs. The complete source remains in the repository as reproducibility evidence, but it will not be printed in the thesis.",
    ))

    parts.extend([
        "# Consolidated correction appendices",
        "",
        "## Appendix A: Complete variable, coefficient and operating-range register",
        "",
        md_table(["Variable", "Meaning", "Transformation", "Fitted domain", "Preferred", "Managed", "Critical", "Decision"], variable_rows),
        "",
        md_table(["Coefficient", "Estimate", "Restriction", "Verification", "Standard error", "95% confidence interval"], coefficient_rows),
        "",
        "The estimated coefficient signs align with the modeled theory: higher attack and privacy pressure reduce Y when controls are fixed, whereas stronger defense and broader participation increase Y. All sign restrictions were satisfied and maximum constraint violation was zero.",
        "",
        "## Appendix B: Fixed federated values and optimized controller values",
        "",
        md_table(["Setting", "Value/search space", "Role", "Interpretation"], federation_rows),
        "",
        "The executable defense mapping is adversarial fraction=0.50DM, training attack epsilon=0.05DM, and Gaussian gradient-noise standard deviation=0.0003DM. These are controller settings and do not constitute a formal differential-privacy guarantee.",
        "",
        "## Appendix C: Data provenance and architecture limitations",
        "",
        md_table(["Data", "Provenance", "Source/generator", "Quantity", "Representation", "Split/design", "Qualification"], data_rows),
        "",
        md_table(["Area", "What is implemented", "Boundary that must be disclosed"], architecture_limitations),
        "",
        "## Appendix D: Proposed objective-to-literature alignment matrix",
        "",
        "**Verification condition:** The exact approved Chapter 1 objective wording was not available in the inspected evidence. Confirm and reconcile the labels, numbering and order below against the approved objectives before inserting this matrix into the thesis.",
        "",
        md_table(["Research objective", "Literature theme", "Authoritative basis", "Use in this study"], literature_rows),
        "",
        "## Appendix E: Assumption, delimitation and limitation matrix",
        "",
        md_table(["Statement", "Rationale", "If violated", "Classification"], assumption_rows),
        "",
        "## Appendix F: Manuscript correction register",
        "",
        md_table(["Comment", "Target location", "Correction", "Disposition", "Evidence basis"], correction_rows),
        "",
        "## Appendix G: Oral-defense cheat sheet",
        "",
        "1. **Values/ranges:** AI and PA are modeled pressure inputs; DM and GS are lowest-cost feasible controls; preferred Y is at least 0.70 and managed Y is at least 0.60.",
        "2. **Equation:** X1=AI, X2=PA, X3=DM and X4=GS; these are predictors and controls, not a causal chain.",
        "3. **Dynamic:** Phase 6 takes specified round AI/PA inputs and reselects DM/GS from 63 candidates; the recorded path is fixed, not live Phase-3/4 telemetry.",
        "4. **Federation values:** Five clients, 12 rounds, two local epochs, batch 64, Adam 0.001; only DM and K/GS are dynamically searched.",
        "5. **Architecture:** Synchronous one-process FedAvg simulation using LFW faces and generated fingerprints.",
        "6. **Modalities:** Face and fingerprint are complementary in theory, but this implementation has no validated same-person fusion.",
        "7. **Deliverable:** Auditable Phase 1-5 attack/defense experiment plus a separate reproducible constrained-policy demonstrator.",
        "8. **Video:** A new authentic isolated run must be recorded; logs alone are not the video deliverable.",
        "9. **Diagrams:** Use original project diagrams and distinguish executed flows from design linkages.",
        "10. **Literature:** Reconcile the proposed matrix with the approved Chapter 1 objectives, then organize Chapter Two by the approved wording, synthesis, limitation, gap and response.",
        "11. **Table 3:** Label every value as literature-derived, measured, fitted or researcher-defined.",
        "12. **Numbering:** Use chapter-based automatic captions and cross-references throughout.",
        "13. **Provenance:** Faces are secondary LFW; fingerprints and controller observations are locally generated.",
        "14. **Low results:** Report synchronized values honestly and state limitations; improvement requires a new experiment.",
        "15. **Assumptions:** Explain necessity and violation consequence; separate delimitations and limitations.",
        "16. **Equations:** Use editable Equation Editor/LaTeX equations with chapter numbering.",
        "17. **Code:** Put algorithms in the thesis and full source in the versioned repository.",
        "",
        "## Appendix H: Claim-to-code-and-artifact traceability",
        "",
        md_table(["Claim/component", "Source file", "Function/mechanism", "Primary artifact"], trace_rows),
        "",
        "### Recorded runtime registry",
        "",
        md_table(["Phase", "Script", "Elapsed seconds", "Execution status"], [[row["phase"], row["script"], f"{float(row['elapsed_seconds']):.6f}", row["status"]] for row in recorded.runtime_rows]),
        "",
        "## Appendix I: Thesis-ready pseudocode for the six simulation modules",
        "",
        "### Algorithm 4.1: Data preparation and client allocation",
        "",
        "**Input:** LFW selection criteria, fingerprint generator seed, image sizes, split ratio and client count.",
        "",
        "**Output:** Standardized face/fingerprint arrays, labels, train/test splits and five client partitions.",
        "",
        "1. Download or load cached LFW faces that satisfy the minimum-images rule.",
        "2. Resize faces, flatten pixels, standardize features and encode face labels.",
        "3. For each synthetic fingerprint identity and impression, generate a ridge pattern, add seeded variation, normalize and label it.",
        "4. Create stratified 80/20 train/test splits for each modality.",
        "5. Allocate training identities to five persistent client partitions.",
        "6. Save arrays, metadata, client files and data-summary figures.",
        "",
        "### Algorithm 4.2: Local training and sample-weighted FedAvg",
        "",
        "**Input:** Prepared modality arrays; K=5; rounds=12; local epochs=2; batch size=64.",
        "",
        "**Output:** Baseline global model, training history and baseline metrics.",
        "",
        "1. Create equal paired batching rows and split them across five in-memory logical clients.",
        "2. Initialize the global dual-head model.",
        "3. For each global round, send the current weight arrays to every logical client.",
        "4. Train each client model locally for two epochs and return its weights and row count.",
        "5. Compute each client weight n_k divided by the total participating row count.",
        "6. Aggregate corresponding parameter arrays and replace the global model weights.",
        "7. Evaluate the global face and fingerprint heads and save the model/history.",
        "",
        "### Algorithm 4.3: Adversarial-attack evaluation",
        "",
        "**Input:** Baseline model, prepared test arrays, epsilon sweep, membership threshold and poison rate.",
        "",
        "**Output:** FGSM/PGD ASR curves, membership results, poisoning results and attack artifacts.",
        "",
        "1. Record clean predictions and identify eligible clean-correct examples for conditional evasion ASR.",
        "2. For every epsilon, generate FGSM perturbations, predict again and compute successful attacks over the eligible denominator.",
        "3. For every epsilon, iteratively generate PGD perturbations inside the permitted bound and compute conditional ASR.",
        "4. Compare member and non-member confidence against the declared threshold and compute membership accuracy/pressure evidence.",
        "5. Flip the declared share of training labels, retrain for the declared poison epochs and record the poisoned result.",
        "6. Save exact parameters, denominators, dictionaries and figures.",
        "",
        "### Algorithm 4.4: Defense-model training and comparison",
        "",
        "**Input:** Phase-1 prepared arrays and defense configurations.",
        "",
        "**Output:** No-defense, DP-style, adversarial-training and combined-model results; final model.",
        "",
        "1. Create four new centrally trained comparison models from the prepared arrays.",
        "2. Train the no-defense condition with standard batches.",
        "3. Train the DP-style condition with Gaussian gradient noise; label it DP-style because clipping and a privacy accountant/formal epsilon-delta accounting are absent.",
        "4. Train the adversarial condition using generated adversarial examples.",
        "5. Train the combined DO-FMBS condition using the declared defense settings.",
        "6. Evaluate clean face accuracy and conditional attack success for every condition.",
        "7. Save the final model and comparison artifacts.",
        "",
        "### Algorithm 4.5: Validation and operational evidence",
        "",
        "**Input:** Baseline/final models, test arrays, epsilon values, repeat count and operational evaluator settings.",
        "",
        "**Output:** Classification, robustness, statistics, claimed-identity FAR/FRR, PA and runtime evidence.",
        "",
        "1. In Phase 5, evaluate baseline and final face classification on the same validation subset.",
        "2. Sweep FGSM epsilon and record classification accuracy/ASR for each model.",
        "3. Repeat seeded subset evaluations and compute the declared statistical comparison.",
        "4. In the post-run evaluator, calibrate a claimed-identity face-verification threshold on the calibration identities.",
        "5. Freeze the threshold, enumerate genuine and impostor claims, and compute FAR/FRR with exact denominators and intervals.",
        "6. Evaluate membership pressure PA and record evaluation runtime.",
        "7. Save all tables, manifests, hashes and figures.",
        "",
        "### Algorithm 4.6: Constrained DM/GS policy selection",
        "",
        "**Input:** Seeded 700-row controller design, sign constraints, fixed 12-round AI/PA path and cost weights.",
        "",
        "**Output:** Constrained coefficients and one accepted/deferred policy for every controller round.",
        "",
        "1. Generate the balanced AI/PA/DM/GS factorial observations using the fixed seed.",
        "2. Estimate B0, B_AI, B_PA, B_DM and B_GS by SLSQP under the four sign restrictions.",
        "3. For each specified round input, enumerate 21 DM values and K in {3,4,5}, giving 63 candidates.",
        "4. Reject any candidate outside the fitted domain and predict Y for every remaining candidate.",
        "5. Among candidates with Y at least 0.70, select the minimum-cost policy.",
        "6. If the preferred set is empty, select the minimum-cost candidate with Y at least 0.60.",
        "7. If both feasible sets are empty, return reject/defer; otherwise record DM, GS, K, Y, cost and status.",
        "8. Save fit, constraint, stress-test and dynamic-schedule artifacts. Do not imply automatic neural-model retraining.",
        "",
        "### Evidence roots",
        "",
        f"- Synchronized neural and controller replay: `{recorded.run_dir}`",
        f"- Operational biometric evaluator: `{recorded.evaluator_dir}`",
        "- Phase-6 model evidence: `simulation_evidence/constrained_dynamic_model/`",
        "- Public source repository: https://github.com/lydiahmachora-del/DO-FMBS-Lydiah",
        "",
        "# References",
        "",
        "1. Dwork, C., and Roth, A. (2014). *The Algorithmic Foundations of Differential Privacy*. Foundations and Trends in Theoretical Computer Science, 9(3-4), 211-407. https://doi.org/10.1561/0400000042",
        "2. Goodfellow, I. J., Shlens, J., and Szegedy, C. (2015). *Explaining and Harnessing Adversarial Examples*. ICLR. https://arxiv.org/abs/1412.6572",
        "3. Huang, G. B., Ramesh, M., Berg, T., and Learned-Miller, E. (2007). *Labeled Faces in the Wild: A Database for Studying Face Recognition in Unconstrained Environments*. University of Massachusetts Amherst, Technical Report 07-49. https://people.cs.umass.edu/~elm/papers/lfw.pdf",
        "4. ISO/IEC. (2021). *ISO/IEC 19795-1:2021 Information technology - Biometric performance testing and reporting - Part 1: Principles and framework*. https://www.iso.org/standard/73515.html",
        "5. Madry, A., Makelov, A., Schmidt, L., Tsipras, D., and Vladu, A. (2018). *Towards Deep Learning Models Resistant to Adversarial Attacks*. ICLR. https://openreview.net/forum?id=rJzIBfZAb",
        "6. McMahan, H. B., Moore, E., Ramage, D., Hampson, S., and Aguera y Arcas, B. (2017). *Communication-Efficient Learning of Deep Networks from Decentralized Data*. Proceedings of AISTATS, PMLR 54, 1273-1282. https://proceedings.mlr.press/v54/mcmahan17a.html",
        "7. Hicklin, R. A., Kiebuzinski, G. I., and Taylor, M. H. (2021). *Best Practices in the Collection and Use of Biometric and Forensic Datasets*. NIST IR 8361. https://doi.org/10.6028/NIST.IR.8361",
        "8. Ross, A., and Jain, A. (2003). Information fusion in biometrics. *Pattern Recognition Letters*, 24(13), 2115-2125. https://doi.org/10.1016/S0167-8655(03)00079-5",
        "9. Shokri, R., Stronati, M., Song, C., and Shmatikov, V. (2017). Membership inference attacks against machine learning models. *IEEE Symposium on Security and Privacy*. https://www.ieee-security.org/TC/SP2017/papers/313.pdf",
        "10. Kraft, D. (1988). *A Software Package for Sequential Quadratic Programming*. DFVLR-FB 88-28, German Aerospace Center. The implementation uses SciPy's documented SLSQP interface: https://docs.scipy.org/doc/scipy/reference/optimize.minimize-slsqp.html",
        "",
        "# Final response position",
        "",
        "The defensible position is that DO-FMBS is a reproducible research prototype with a transparent constrained-policy mechanism. Its evidence supports the implemented face-classification, claimed-identity face-verification, attack/defense and synthetic controller experiments within their declared scopes. It does not yet establish real co-enrolled multimodal fusion, formal differential privacy, a physical federated deployment or production-grade resilience at high attack intensity. Those boundaries strengthen the credibility of the correction because they separate demonstrated results from future work.",
    ])

    return "\n".join(parts).strip() + "\n"


def markdown_for_file(markdown: str) -> str:
    lines: list[str] = []
    for line in markdown.splitlines():
        match = FIGURE_RE.match(line.strip())
        if match:
            filename, caption = match.groups()
            lines.extend([f"![{caption}]({filename})", "", f"*{caption}*"])
        else:
            lines.append(line)
    return "\n".join(lines).strip() + "\n"


def ascii_safe(text: str) -> str:
    replacements = {
        "≥": ">=", "≤": "<=", "→": "->", "–": "-", "—": "-",
        "‘": "'", "’": "'", "“": '"', "”": '"', "×": "x", "ε": "epsilon",
    }
    value = _plain_markdown(text)
    for source, target in replacements.items():
        value = value.replace(source, target)
    return value.encode("latin-1", "replace").decode("latin-1")


def split_urls(text: str):
    """Yield plain-text and URL pieces, excluding sentence punctuation."""
    cursor = 0
    for match in URL_RE.finditer(text):
        if match.start() > cursor:
            yield False, text[cursor:match.start()]
        raw = match.group(0)
        url = raw
        while url and url[-1] in ".,;:":
            # Preserve punctuation that is structurally part of a URL, such
            # as the final `.pdf` or a DOI's `(03)00079-5` suffix. Sentence
            # punctuation is removed only when it is not part of those forms.
            if url.endswith(".pdf") or (url.endswith(")") and "(" in url):
                break
            url = url[:-1]
        yield True, url
        if len(url) < len(raw):
            yield False, raw[len(url):]
        cursor = match.end()
    if cursor < len(text):
        yield False, text[cursor:]


def write_docx(markdown: str, path: Path, figure_dir: Path) -> None:
    from docx import Document
    from docx.enum.table import WD_CELL_VERTICAL_ALIGNMENT
    from docx.enum.text import WD_ALIGN_PARAGRAPH
    from docx.oxml import OxmlElement
    from docx.oxml.ns import qn
    from docx.shared import Inches, Pt, RGBColor

    def add_hyperlink(paragraph, url: str, label: str | None = None) -> None:
        """Append a standard external hyperlink relationship to a paragraph."""
        relationship_id = paragraph.part.relate_to(
            url,
            "http://schemas.openxmlformats.org/officeDocument/2006/relationships/hyperlink",
            is_external=True,
        )
        hyperlink = OxmlElement("w:hyperlink")
        hyperlink.set(qn("r:id"), relationship_id)
        run = OxmlElement("w:r")
        properties = OxmlElement("w:rPr")
        color = OxmlElement("w:color")
        color.set(qn("w:val"), "0563C1")
        underline = OxmlElement("w:u")
        underline.set(qn("w:val"), "single")
        properties.extend((color, underline))
        run.append(properties)
        value = OxmlElement("w:t")
        value.text = label or url
        run.append(value)
        hyperlink.append(run)
        paragraph._p.append(hyperlink)

    def add_text_with_links(paragraph, text: str) -> None:
        for is_url, value in split_urls(_plain_markdown(text)):
            if is_url:
                add_hyperlink(paragraph, value)
            elif value:
                paragraph.add_run(value)

    def linkify_paragraph(paragraph) -> None:
        """Convert visible URL runs in list/reference paragraphs to links."""
        text = paragraph.text
        if not URL_RE.search(text):
            return
        for child in list(paragraph._p):
            if child.tag != qn("w:pPr"):
                paragraph._p.remove(child)
        add_text_with_links(paragraph, text)

    document = Document()
    section = document.sections[0]
    section.top_margin = Inches(0.68)
    section.bottom_margin = Inches(0.68)
    section.left_margin = Inches(0.70)
    section.right_margin = Inches(0.70)
    document.core_properties.title = "DO-FMBS Response to Mock-Presentation Comments"
    document.core_properties.subject = "Formal response to 17 examiner comments"
    document.core_properties.author = "Lydiah Moraa Machora"
    document.core_properties.keywords = "DO-FMBS, examiner response, mock presentation, corrections"

    styles = document.styles
    styles["Normal"].font.name = "Times New Roman"
    styles["Normal"].font.size = Pt(10.2)
    for name, size, color in (
        ("Title", 21, "17365D"), ("Heading 1", 16, "17365D"),
        ("Heading 2", 13, "2F5597"), ("Heading 3", 10.5, "385723"),
    ):
        styles[name].font.name = "Times New Roman"
        styles[name].font.size = Pt(size)
        styles[name].font.color.rgb = RGBColor.from_string(color)

    first_heading = True
    equation_buffers = []
    for kind, payload in _markdown_blocks(markdown):
        if kind == "heading1" and first_heading:
            paragraph = document.add_paragraph(_plain_markdown(str(payload)), style="Title")
            paragraph.alignment = WD_ALIGN_PARAGRAPH.CENTER
            paragraph.paragraph_format.space_before = Pt(72)
            paragraph.paragraph_format.space_after = Pt(20)
            first_heading = False
        elif kind in {"heading1", "heading2", "heading3"}:
            level = {"heading1": 1, "heading2": 1, "heading3": 2}[kind]
            text = _plain_markdown(str(payload))
            paragraph = document.add_heading(text, level=level)
            paragraph.paragraph_format.keep_with_next = True
            if kind == "heading1" or re.match(r"Response to Comment (?:1|5|9|13|17):", text):
                paragraph.paragraph_format.page_break_before = True
        elif kind == "table":
            rows = payload
            table = document.add_table(rows=len(rows), cols=len(rows[0]))
            table.style = "Table Grid"
            for row_index, row in enumerate(rows):
                for column_index, value in enumerate(row):
                    cell = table.cell(row_index, column_index)
                    cell.text = _plain_markdown(str(value))
                    cell.vertical_alignment = WD_CELL_VERTICAL_ALIGNMENT.CENTER
                    if row_index == 0:
                        shading = OxmlElement("w:shd")
                        shading.set(qn("w:fill"), "17365D")
                        cell._tc.get_or_add_tcPr().append(shading)
                    for paragraph in cell.paragraphs:
                        paragraph.paragraph_format.space_after = Pt(1.5)
                        for run in paragraph.runs:
                            run.font.name = "Times New Roman"
                            run.font.size = Pt(6.6 if len(rows[0]) >= 7 else 7.5)
                            if row_index == 0:
                                run.bold = True
                                run.font.color.rgb = RGBColor(255, 255, 255)
            document.add_paragraph().paragraph_format.space_after = Pt(1)
        elif kind == "equation":
            paragraph = document.add_paragraph()
            paragraph.alignment = WD_ALIGN_PARAGRAPH.CENTER
            buffer = _equation_image_buffer(str(payload))
            equation_buffers.append(buffer)
            from PIL import Image as PILImage
            with PILImage.open(buffer) as image:
                ratio = image.width / max(image.height, 1)
            buffer.seek(0)
            width = min(6.4, max(2.5, ratio * 0.32))
            paragraph.add_run().add_picture(buffer, width=Inches(width))
        elif kind in {"bullet", "number"}:
            text = payload[1] if kind == "number" else payload
            paragraph = document.add_paragraph(_plain_markdown(str(text)), style="List Number" if kind == "number" else "List Bullet")
            paragraph.paragraph_format.space_after = Pt(3)
        else:
            text = str(payload)
            match = FIGURE_RE.match(text.strip())
            if match:
                filename, caption = match.groups()
                figure_path = figure_dir / filename
                paragraph = document.add_paragraph()
                paragraph.alignment = WD_ALIGN_PARAGRAPH.CENTER
                paragraph.add_run().add_picture(str(figure_path), width=Inches(6.35))
                caption_paragraph = document.add_paragraph(caption, style="Caption")
                caption_paragraph.alignment = WD_ALIGN_PARAGRAPH.CENTER
            elif text == "---":
                paragraph = document.add_paragraph()
                border = OxmlElement("w:pBdr")
                bottom = OxmlElement("w:bottom")
                bottom.set(qn("w:val"), "single")
                bottom.set(qn("w:sz"), "6")
                bottom.set(qn("w:color"), "B4C6E7")
                border.append(bottom)
                paragraph._p.get_or_add_pPr().append(border)
            else:
                paragraph = document.add_paragraph()
                add_text_with_links(paragraph, text)
                paragraph.paragraph_format.space_after = Pt(5)
                paragraph.paragraph_format.line_spacing = 1.08

    header = section.header.paragraphs[0]
    header.text = "DO-FMBS | Formal Response to Mock-Presentation Comments"
    header.alignment = WD_ALIGN_PARAGRAPH.CENTER
    for run in header.runs:
        run.font.name = "Times New Roman"
        run.font.size = Pt(8)
        run.font.color.rgb = RGBColor.from_string("666666")

    footer = section.footer.paragraphs[0]
    footer.text = "Lydiah Moraa Machora | I152/P/9515/23 | 2026 | Page "
    page_run = footer.add_run()
    field_begin = OxmlElement("w:fldChar")
    field_begin.set(qn("w:fldCharType"), "begin")
    field_code = OxmlElement("w:instrText")
    field_code.set(qn("xml:space"), "preserve")
    field_code.text = " PAGE "
    field_end = OxmlElement("w:fldChar")
    field_end.set(qn("w:fldCharType"), "end")
    page_run._r.extend((field_begin, field_code, field_end))
    footer.alignment = WD_ALIGN_PARAGRAPH.CENTER
    # Numbered references and bulleted evidence-root entries are created by
    # the list branches above. Convert their visible URLs to true Word links.
    for paragraph in document.paragraphs:
        linkify_paragraph(paragraph)
    document.save(path)


def write_pdf(markdown: str, path: Path, figure_dir: Path) -> None:
    from reportlab.lib import colors
    from reportlab.lib.enums import TA_CENTER
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
    from reportlab.lib.units import mm
    from reportlab.platypus import CondPageBreak, Image, PageBreak, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle

    styles = getSampleStyleSheet()
    styles.add(ParagraphStyle(name="ERTitle", parent=styles["Title"], fontName="Times-Bold", fontSize=19, leading=23, textColor=colors.HexColor("#17365D"), alignment=TA_CENTER, spaceAfter=10))
    styles.add(ParagraphStyle(name="ERH1", parent=styles["Heading1"], fontName="Times-Bold", fontSize=14.5, leading=17, textColor=colors.HexColor("#17365D"), spaceBefore=8, spaceAfter=5, keepWithNext=True))
    styles.add(ParagraphStyle(name="ERH2", parent=styles["Heading2"], fontName="Times-Bold", fontSize=11.5, leading=14, textColor=colors.HexColor("#2F5597"), spaceBefore=6, spaceAfter=4, keepWithNext=True))
    styles.add(ParagraphStyle(name="ERBody", parent=styles["BodyText"], fontName="Times-Roman", fontSize=9.2, leading=12.6, alignment=4, spaceAfter=5))
    styles.add(ParagraphStyle(name="ERCaption", parent=styles["BodyText"], fontName="Times-Italic", fontSize=8, leading=10, alignment=TA_CENTER, spaceAfter=7))
    styles.add(ParagraphStyle(name="ERCell", parent=styles["BodyText"], fontName="Times-Roman", fontSize=6.4, leading=7.8))

    def pdf_markup(text: str) -> str:
        pieces: list[str] = []
        for is_url, value in split_urls(ascii_safe(text)):
            escaped = html_escape(value)
            if is_url:
                pieces.append(f'<link href="{escaped}" color="#0563C1"><u>{escaped}</u></link>')
            else:
                pieces.append(escaped)
        return "".join(pieces)

    def footer(canvas, document):
        canvas.saveState()
        page_width, _ = A4
        canvas.setStrokeColor(colors.HexColor("#B4C6E7"))
        canvas.line(14 * mm, 11 * mm, page_width - 14 * mm, 11 * mm)
        canvas.setFont("Times-Roman", 7.3)
        canvas.setFillColor(colors.HexColor("#666666"))
        canvas.drawString(14 * mm, 7 * mm, "DO-FMBS | Response to Mock-Presentation Comments")
        canvas.drawRightString(page_width - 14 * mm, 7 * mm, f"Page {document.page}")
        canvas.restoreState()

    class CompactResponseDocTemplate(SimpleDocTemplate):
        """Suppress page breaks that would create a nearly empty orphan page.

        The response intentionally starts major examiner groups on new pages,
        except when the current page is already substantially empty.  In that
        case a forced break would leave only a short carry-over paragraph on
        the preceding page, so the break is omitted.
        """

        def handle_pageBreak(self, slow=None):
            frame = getattr(self, "frame", None)
            if frame is not None:
                used = frame._y1 - frame._y
                if used < 38 * mm:
                    return
            super().handle_pageBreak(slow)

    document = CompactResponseDocTemplate(
        str(path), pagesize=A4, leftMargin=14 * mm, rightMargin=14 * mm,
        topMargin=15 * mm, bottomMargin=15 * mm,
        title="DO-FMBS Response to Mock-Presentation Comments",
        author="Lydiah Moraa Machora",
    )
    story = []
    first_heading = True
    equation_buffers = []
    for kind, payload in _markdown_blocks(markdown):
        if kind == "heading1" and first_heading:
            story.extend([Spacer(1, 35 * mm), Paragraph(html_escape(ascii_safe(str(payload))), styles["ERTitle"]), Spacer(1, 8 * mm)])
            first_heading = False
        elif kind in {"heading1", "heading2"}:
            title = str(payload)
            if kind == "heading1" or re.match(r"Response to Comment (?:1|5|9|13|17):", title):
                story.append(PageBreak())
            story.append(Paragraph(html_escape(ascii_safe(title)), styles["ERH1"]))
        elif kind == "heading3":
            story.append(CondPageBreak(24 * mm))
            story.append(Paragraph(html_escape(ascii_safe(str(payload))), styles["ERH2"]))
        elif kind == "table":
            rows = [[Paragraph(html_escape(ascii_safe(str(cell))), styles["ERCell"]) for cell in row] for row in payload]
            width = A4[0] - 31 * mm
            columns = len(rows[0])
            table = Table(rows, colWidths=[width / columns] * columns, repeatRows=1, hAlign="CENTER")
            table.setStyle(TableStyle([
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#17365D")),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                ("GRID", (0, 0), (-1, -1), 0.25, colors.HexColor("#A6A6A6")),
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#F2F6FA")]),
                ("LEFTPADDING", (0, 0), (-1, -1), 2.2),
                ("RIGHTPADDING", (0, 0), (-1, -1), 2.2),
                ("TOPPADDING", (0, 0), (-1, -1), 2.5),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 2.5),
            ]))
            story.extend([table, Spacer(1, 5)])
        elif kind == "equation":
            buffer = _equation_image_buffer(str(payload))
            equation_buffers.append(buffer)
            image = Image(buffer)
            image._restrictSize(A4[0] - 30 * mm, 18 * mm)
            image.hAlign = "CENTER"
            story.extend([image, Spacer(1, 4)])
        else:
            if kind == "bullet":
                value = "- " + str(payload)
            elif kind == "number":
                value = f"{payload[0]}. {payload[1]}"
            else:
                value = str(payload)
            match = FIGURE_RE.match(value.strip())
            if match:
                filename, caption = match.groups()
                story.append(CondPageBreak(110 * mm))
                image = Image(str(figure_dir / filename))
                image._restrictSize(A4[0] - 30 * mm, 112 * mm)
                image.hAlign = "CENTER"
                story.extend([image, Paragraph(html_escape(ascii_safe(caption)), styles["ERCaption"])])
            elif value == "---":
                story.append(Spacer(1, 5))
            else:
                story.append(Paragraph(pdf_markup(value), styles["ERBody"]))
    document.build(story, onFirstPage=footer, onLaterPages=footer)


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


def validate_text(text: str, label: str) -> None:
    lowered = text.casefold()
    required = (
        "response to comment 1", "response to comment 17", "evidence declaration",
        "x1", "b_ai", "0.400738", "7.852%", "11.146%", "1522.307974",
        "labeled faces in the wild", "generated locally", "action required",
        "objective-to-literature", "assumption", "oral-defense cheat sheet",
        "traceability", "not yet establish real co-enrolled multimodal fusion",
    )
    for phrase in required:
        if phrase not in lowered:
            raise ValueError(f"{label} is missing required content: {phrase}")
    for number_value in range(1, 18):
        if f"response to comment {number_value}" not in lowered:
            raise ValueError(f"{label} is missing response {number_value}")


def generate(args: argparse.Namespace) -> list[Path]:
    evidence = load_evidence(args)
    required_sources = [
        ROOT / "phase1_data_preparation.py",
        ROOT / "phase2_federated_learning.py",
        ROOT / "phase3_adversarial_attacks.py",
        ROOT / "phase4_defense_mechanisms.py",
        ROOT / "phase5_model_validation.py",
        ROOT / "phase6_constrained_dynamic_optimization.py",
        ROOT / "scripts" / "constrained_dynamic_model.py",
        ROOT / "scripts" / "measure_operational_biometric_metrics.py",
        ROOT / "scripts" / "run_isolated_instrumented_pipeline.py",
    ]
    for source in required_sources:
        if not source.is_file() or source.stat().st_size == 0:
            raise FileNotFoundError(f"Required response source is missing: {source}")
    if args.validate_inputs_only:
        return []

    formats = {item.strip().lower() for item in args.formats.split(",") if item.strip()}
    formats.add("md")
    if formats - {"md", "docx", "pdf"}:
        raise ValueError("Formats must be md, docx and/or pdf")

    evidence.output_dir.mkdir(parents=True, exist_ok=True)
    figure_path = evidence.output_dir / FIGURE_NAME
    if not figure_path.is_file() or figure_path.stat().st_size < 10_000:
        raise FileNotFoundError(f"Required original response figure is missing or incomplete: {figure_path}")

    generated_at = datetime.now(ZoneInfo("Africa/Nairobi")).isoformat(timespec="seconds")
    conversion_markdown = build_markdown(evidence, generated_at)
    file_markdown = markdown_for_file(conversion_markdown)
    validate_text(file_markdown, "Markdown response")

    outputs: list[Path] = []
    md_path = evidence.output_dir / f"{REPORT_STEM}.md"
    md_path.write_text(file_markdown, encoding="utf-8")
    outputs.append(md_path)
    if "docx" in formats:
        docx_path = evidence.output_dir / f"{REPORT_STEM}.docx"
        write_docx(conversion_markdown, docx_path, evidence.output_dir)
        validate_text(extract_docx_text(docx_path), "DOCX response")
        outputs.append(docx_path)
    if "pdf" in formats:
        pdf_path = evidence.output_dir / f"{REPORT_STEM}.pdf"
        write_pdf(conversion_markdown, pdf_path, evidence.output_dir)
        validate_text(extract_pdf_text(pdf_path), "PDF response")
        outputs.append(pdf_path)

    sources = [*required_sources, *evidence.source_paths]
    unique_sources = list(dict.fromkeys(path.resolve() for path in sources))
    from PIL import Image as PILImage
    with PILImage.open(figure_path) as image:
        figure_width, figure_height = image.size

    manifest = {
        "generated_at_africa_nairobi": generated_at,
        "status": "FORMAL-EXAMINER-RESPONSE-GENERATED-FROM-SYNCHRONIZED-EVIDENCE",
        "comments_answered": 17,
        "response_blocks_per_comment": 5,
        "synchronized_replay": evidence.recorded.run_dir.name,
        "operational_evaluator": evidence.recorded.evaluator_dir.name,
        "evidence_declaration": "Phase 1-5 biometric evidence and Phase-6 seeded synthetic controller evidence are reported with separate scopes.",
        "outstanding_actions": [
            "Record an authentic narrated end-to-end demonstration.",
            "Complete the whole-thesis literature, numbering, caption and code-to-pseudocode audits.",
            "Verify the exact approved Chapter 1 objectives and reconcile the proposed objective-to-literature matrix before insertion.",
            "Verify the exact Table 3 before inserting its row-specific rationale.",
            "Insert the proposed evidence-scoped metrics into the source abstract after supervisor review.",
            "Revise Chapter One using the supplied assumption, delimitation and limitation matrix.",
            "Correct Phase-1 preprocessing to fit scalers on training partitions only, then rerun dependent phases and refresh reported metrics.",
            "Treat co-enrolled multimodal fusion, formal differential privacy and physical federation as future experimental work.",
        ],
        "figure": {
            "path": figure_path.name,
            "bytes": figure_path.stat().st_size,
            "sha256": sha256(figure_path),
            "width_px": figure_width,
            "height_px": figure_height,
        },
        "sources": [
            {
                "path": path.relative_to(ROOT).as_posix() if path.is_relative_to(ROOT) else str(path),
                "bytes": path.stat().st_size,
                "sha256": sha256(path),
            }
            for path in unique_sources
        ],
        "documents": [
            {"path": path.name, "bytes": path.stat().st_size, "sha256": sha256(path)}
            for path in outputs
        ],
    }
    manifest_path = evidence.output_dir / "DO_FMBS_RESPONSE_TO_MOCK_PRESENTATION_COMMENTS_MANIFEST.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    outputs.append(manifest_path)
    return outputs


def main() -> None:
    args = parse_args()
    outputs = generate(args)
    if args.validate_inputs_only:
        print("Examiner-response inputs validated successfully.")
        return
    print("Generated formal response package:")
    for output in outputs:
        print(f"- {output}")


if __name__ == "__main__":
    main()
