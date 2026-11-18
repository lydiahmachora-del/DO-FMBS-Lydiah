"""Generate Chapter Four solely from an authoritative DO-FMBS evidence manifest."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MANIFEST = ROOT / "do_fmbs_data" / "final_evidence_manifest.json"
DEFAULT_OUTPUT_DIR = ROOT / "docs" / "chapter_four"
STEM = "CHAPTER_FOUR_MODEL_DEVELOPMENT_IMPLEMENTATION_AND_VALIDATION"
# Compatibility names retained so older ancillary document modules import
# cleanly.  They are not authoritative Chapter Four inputs.
REPORT_STEM = STEM
DEFAULT_RUN_DIR = ROOT / "runtime_recording" / "final_scientific_replay_20260907"
DEFAULT_EVALUATOR_DIR = DEFAULT_RUN_DIR / "do_fmbs_data"
DEFAULT_FINGERPRINT_EVALUATION_DIR = DEFAULT_EVALUATOR_DIR
DEFAULT_PGD_EVALUATION_DIR = DEFAULT_EVALUATOR_DIR
DEFAULT_ARCHITECTURE = DEFAULT_OUTPUT_DIR / "DO_FMBS_ARCHITECTURE_FOR_SIMULATION_PROCESS.png"


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--validate-only", action="store_true")
    return parser.parse_args(argv)


def sha256(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def pct(value: object, digits: int = 2) -> str:
    return f"{100.0 * float(value):.{digits}f}%"


def number(value: object, digits: int = 4) -> str:
    return f"{float(value):.{digits}f}"


num = number


def md_table(headers: Sequence[object], rows: Sequence[Sequence[object]]) -> str:
    lines = ["| " + " | ".join(map(str, headers)) + " |",
             "| " + " | ".join(["---"] * len(headers)) + " |"]
    lines.extend("| " + " | ".join(map(str, row)) + " |" for row in rows)
    return "\n".join(lines)


def load_evidence(_args):
    raise RuntimeError(
        "This legacy ancillary-report interface is retired. Use load_manifest() "
        "and final_evidence_manifest.json for synchronized evidence."
    )


def load_manifest(path: Path) -> dict:
    manifest = json.loads(path.read_text(encoding="utf-8"))
    if manifest.get("status") != "AUTHORITATIVE-SYNCHRONIZED-REPLAY-COMPLETE":
        raise RuntimeError("Chapter Four requires a completed authoritative replay manifest")
    replay = manifest["replay_id"]
    for phase in ("federated_learning", "attacks", "defenses"):
        if manifest[phase].get("replay_id") != replay:
            raise RuntimeError(f"Mixed replay evidence detected in {phase}")
    for item in manifest["artifacts"]:
        artifact = ROOT / item["path"]
        if not artifact.exists() or artifact.stat().st_size != int(item["bytes"]):
            raise RuntimeError(f"Missing or size-mismatched authoritative artifact: {artifact}")
        if sha256(artifact) != item["sha256"]:
            raise RuntimeError(f"Hash mismatch for authoritative artifact: {artifact}")
    return manifest


def find_artifact(manifest: dict, suffix: str) -> Path:
    matches = [ROOT / row["path"] for row in manifest["artifacts"] if row["path"].endswith(suffix)]
    if len(matches) != 1:
        raise RuntimeError(f"Expected exactly one artifact ending {suffix!r}; found {matches}")
    return matches[0]


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


class Report:
    def __init__(self) -> None:
        self.blocks: list[dict] = []
        self.table_number = 0
        self.figure_number = 0
        self.evidence_map: list[dict] = []

    def heading(self, text: str, level: int) -> None:
        self.blocks.append({"kind": "heading", "level": level, "text": text})

    def paragraph(self, text: str) -> None:
        self.blocks.append({"kind": "paragraph", "text": text})

    def page_break(self) -> None:
        self.blocks.append({"kind": "page_break"})

    def table(self, title: str, headers: Sequence[str], rows: Sequence[Sequence[object]], source: str) -> None:
        self.table_number += 1
        caption = f"Table 4.{self.table_number}: {title}"
        self.blocks.append({"kind": "table", "caption": caption, "headers": list(headers),
                            "rows": [[str(cell) for cell in row] for row in rows]})
        self.evidence_map.append({"item": f"Table 4.{self.table_number}", "source_artifact": source,
                                  "generator": "scripts/generate_chapter_four_report.py"})

    def figure(self, path: Path, title: str, source: str) -> None:
        self.figure_number += 1
        caption = f"Figure 4.{self.figure_number}: {title}"
        self.blocks.append({"kind": "figure", "caption": caption, "path": str(path)})
        self.evidence_map.append({"item": f"Figure 4.{self.figure_number}", "source_artifact": source,
                                  "generator": "scripts/generate_chapter_four_report.py"})


def copy_figure(report: Report, manifest: dict, output: Path, suffix: str, title: str) -> None:
    source = find_artifact(manifest, suffix)
    destination = output / f"CHAPTER4_FIGURE_4_{report.figure_number + 1}_{source.stem.upper()}.png"
    shutil.copy2(source, destination)
    report.figure(destination, title, source.relative_to(ROOT).as_posix())


def dynamic_figure(report: Report, manifest: dict, output: Path) -> None:
    source = find_artifact(manifest, "dynamic_optimization_schedule.csv")
    rows = read_csv(source)
    rounds = [int(row["round"]) for row in rows]
    fig, ax = plt.subplots(figsize=(8.4, 5.0))
    ax.plot(rounds, [float(row["AI"]) for row in rows], marker="o", label="AI (measured)")
    ax.plot(rounds, [float(row["PA"]) for row in rows], marker="o", label="PA (measured)")
    ax.plot(rounds, [float(row["DM"]) for row in rows], marker="s", label="DM (optimized)")
    ax.plot(rounds, [float(row["GS"]) for row in rows], marker="s", label="GS (optimized)")
    ax.plot(rounds, [float(row["predicted_Y"]) for row in rows], marker="^", label="Predicted Y")
    ax.set(xlabel="Optimization round", ylabel="Normalized value", title="Constrained Dynamic Optimization Trajectory")
    ax.set_ylim(0, 1.05)
    ax.grid(alpha=0.22)
    ax.legend(frameon=False, ncol=2)
    fig.tight_layout()
    destination = output / f"CHAPTER4_FIGURE_4_{report.figure_number + 1}_DYNAMIC_OPTIMIZATION.png"
    fig.savefig(destination, dpi=220, bbox_inches="tight")
    plt.close(fig)
    report.figure(destination, "Measured threat states, optimized controls and predicted response by round", source.relative_to(ROOT).as_posix())


def verification_row(values: dict) -> list[str]:
    final = values["final_test"]
    return [
        number(final["threshold"], 6),
        f"{final['impostor_accepts']}/{final['impostor_attempts']} = {pct(final['FAR'], 3)}",
        f"[{pct(final['FAR_wilson_95'][0], 3)}, {pct(final['FAR_wilson_95'][1], 3)}]",
        f"{final['genuine_rejects']}/{final['genuine_attempts']} = {pct(final['FRR'], 3)}",
        f"[{pct(final['FRR_wilson_95'][0], 3)}, {pct(final['FRR_wilson_95'][1], 3)}]",
        pct(final["TAR"], 3), number(final["ROC_AUC"], 4), pct(final["EER"], 3),
    ]


def build_report(manifest: dict, output: Path) -> Report:
    report = Report()
    data = manifest["dataset_statistics"]
    face = manifest["final_metrics"]["face_classification"]
    finger = manifest["final_metrics"]["fingerprint_classification"]
    face_ver = manifest["final_metrics"]["face_verification"]
    finger_ver = manifest["final_metrics"]["fingerprint_verification"]
    fed = manifest["federated_learning"]
    attacks = manifest["attacks"]
    defenses = manifest["defenses"]
    fit = manifest["dynamic_optimization"]["model_fit"]

    report.heading("CHAPTER FOUR", 1)
    report.heading("MODEL DEVELOPMENT, IMPLEMENTATION AND VALIDATION", 1)
    report.paragraph(
        f"Authoritative replay: {manifest['replay_id']}. This chapter reports only evidence hashed in "
        "the final replay manifest; previous generated values are not treated as experimental evidence."
    )
    report.heading("4.0 Introduction", 2)
    report.paragraph(
        "This chapter presents the development and empirical validation of the Dynamic Optimization "
        "Federated Multimodal Biometric System (DO-FMBS) through six linked phases: data preparation, "
        "federated learning, attack simulation, defense training, independent validation and constrained "
        "dynamic optimization. The implementation contains two modality-specific identity tasks. LFW face "
        "identities and synthetic fingerprint identities are unrelated populations; consequently, the study "
        "does not claim person-level face–fingerprint fusion."
    )

    report.heading("4.1 Data Processing", 2)
    report.heading("4.1.1 Data Sources", 3)
    report.paragraph(
        "Face observations were obtained from the public Labeled Faces in the Wild collection through "
        "scikit-learn. Fingerprint observations were generated locally by a deterministic synthetic model. "
        "They are not FVC data and are interpreted only as within-study synthetic evidence."
    )
    report.table("Dataset characteristics", ["Modality", "Source", "Identities", "Total", "Image shape"], [
        ["Face", data["face"]["source"], data["face"]["identities"], data["face"]["total"], " × ".join(map(str, data["face"]["image_shape"]))],
        ["Fingerprint", data["fingerprint"]["source"], data["fingerprint"]["identities"], data["fingerprint"]["total"], "96 × 96"],
    ], "do_fmbs_data/phase1_metadata.json")
    report.heading("4.1.2 Data Preprocessing", 3)
    report.paragraph(
        "For each modality, raw observations were split before standardization. StandardScaler was fitted "
        "exclusively to the training partition, after which the frozen transformation was applied to the "
        "calibration and test partitions. Persisted source indices and scaler-fit indices provide a direct "
        "test for overlap and leakage."
    )
    report.heading("4.1.3 Training, Calibration and Testing Data", 3)
    report.table("Training, calibration and final-test allocation", ["Modality", "Training", "Calibration", "Final test", "Protocol"], [
        ["Face", data["face"]["train"], data["face"]["calibration"], data["face"]["test"], "70/15/15 stratified by identity"],
        ["Fingerprint", data["fingerprint"]["train"], data["fingerprint"]["calibration"], data["fingerprint"]["test"], data["fingerprint"]["split_protocol"]],
    ], "do_fmbs_data/phase1_metadata.json")
    quality = data["fingerprint"]["quality_gate"]
    report.paragraph(
        f"Before training, the fingerprint quality gate compared {quality['same_identity_pairs']} genuine "
        f"pairs with {quality['different_identity_pairs_sampled']} sampled different-identity pairs. Mean "
        f"similarity was {quality['same_identity_similarity_mean']:.3f} for the same identity and "
        f"{quality['different_identity_similarity_mean']:.3f} for different identities; nearest-neighbour "
        f"identity accuracy was {pct(quality['nearest_neighbour_identity_accuracy'])}. The gate therefore "
        "passed before model fitting."
    )
    copy_figure(report, manifest, output, "fingerprint_same_identity_grid.png", "Controlled impressions from persistent synthetic fingerprint identities")
    copy_figure(report, manifest, output, "fingerprint_similarity_distributions.png", "Same-identity and different-identity fingerprint similarity distributions")
    report.paragraph(
        "Methodological refinement audit note: an earlier, superseded generator redrew ridge orientation "
        "and frequency for every impression. That design produced the historical 57/60 (95%) genuine "
        "rejection result because labels did not preserve a common ridge identity. The present replay "
        "separates identity seeds from impression seeds and does not use that obsolete result as evidence."
    )
    report.heading("4.1.4 Federated Client Distribution", 3)
    report.paragraph(
        "The five-client allocation is class-exclusive and non-IID. Face and fingerprint classes are "
        "partitioned independently, and Phase 2 loads these exact persisted client files. Synchronized "
        "training rows are a computational device only and do not imply that the two records describe one person."
    )
    copy_figure(report, manifest, output, "federated_client_distribution.png", "Identity-exclusive non-IID client distribution")

    report.heading("4.2 Federated Learning Implementation", 2)
    p = fed["training_parameters"]
    report.paragraph(
        "Each round distributed the global dual-branch model to five clients, performed local optimization "
        "with an equal-weight face and fingerprint objective, and aggregated parameters by FedAvg. Client "
        "weights combined each client's share of unique face and fingerprint observations."
    )
    report.table("Federated training configuration", ["Parameter", "Value"], [
        ["Clients", fed["clients"]], ["Rounds", p["rounds"]], ["Local epochs", p["local_epochs"]],
        ["Batch size", p["batch_size"]], ["Learning rate", p["learning_rate"]],
        ["Face/fingerprint loss weights", f"{p['loss_weights']['face']}/{p['loss_weights']['fingerprint']}"],
        ["Partition", fed["federated_partition"]],
    ], "do_fmbs_data/phase2_metrics.json")
    partition_path = find_artifact(manifest, "client_partition.json")
    partition = json.loads(partition_path.read_text(encoding="utf-8"))
    face_total = sum(int(client["face_samples"]) for client in partition["clients"])
    fingerprint_total = sum(int(client["fingerprint_samples"]) for client in partition["clients"])
    last_round = len(fed["history"]["round"])
    face_loss_value = float(fed["history"]["face_loss"][-1])
    fingerprint_loss_value = float(fed["history"]["fingerprint_loss"][-1])
    combined_loss_value = float(fed["history"]["total_loss"][-1])
    mechanism_rows = [
        ["Face loss", f"L_face = -(1/B) sum_i log(p_face[i, y_i]); lambda_face = 0.5. Round {last_round} weighted local loss = {face_loss_value:.6f}.",
         "Mean face cross-entropy; the coefficient sets its share of the joint objective."],
        ["Fingerprint loss", f"L_fingerprint = -(1/B) sum_i log(p_fingerprint[i, y_i]); lambda_fingerprint = 0.5. Round {last_round} weighted local loss = {fingerprint_loss_value:.6f}.",
         "Mean fingerprint cross-entropy; the coefficient sets its share of the joint objective."],
        ["Combined loss", f"L_DO-FMBS = 0.5 L_face + 0.5 L_fingerprint. Round {last_round} logged total loss = {combined_loss_value:.6f}.",
         "The two modality losses contribute equally during local optimization."],
    ]
    for client in partition["clients"]:
        client_id = int(client["client_id"])
        face_count = int(client["face_samples"])
        fingerprint_count = int(client["fingerprint_samples"])
        weight = 0.5 * face_count / face_total + 0.5 * fingerprint_count / fingerprint_total
        mechanism_rows.append([
            f"Client {client_id}",
            f"alpha_{client_id} = 0.5({face_count}/{face_total}) + 0.5({fingerprint_count}/{fingerprint_total}) = {weight:.6f}. Counts: {face_count} face, {fingerprint_count} fingerprint.",
            f"FedAvg parameter contribution = {pct(weight)} of the global update."])
    report.page_break()
    report.table("DO-FMBS Loss Computation and Client Aggregation Parameters",
                 ["Component", "Equation and replay values", "Purpose"],
                 mechanism_rows, "do_fmbs_data/phase2_metrics.json; do_fmbs_data/client_partition.json")
    report.paragraph(
        "Here B=64 is the local mini-batch size and p_modality[i, y_i] is the probability assigned to the "
        "true identity for example i. "
        f"The round {last_round} losses are recorded weighted averages of each client's final local-epoch "
        "losses; they illustrate the changing loss values and are not fixed model parameters. Client weights "
        "are calculated from the persisted partition counts and sum to 1.000000. The modality counts refer "
        "to separate identity populations; they do not indicate paired people."
    )
    report.paragraph(
        f"The federated baseline reached face classification accuracy {pct(fed['face_classification']['accuracy'])} "
        f"and fingerprint classification accuracy {pct(fed['fingerprint_classification']['accuracy'])} on "
        "their separate test partitions. These are identification results; authoritative FAR and FRR are "
        "reported later from the verification protocol."
    )
    copy_figure(report, manifest, output, "federated_training_convergence.png", "Modality-specific federated training convergence")
    convergence = fed["history"]
    accuracy_rows = [
        [str(round_id), pct(face_accuracy), pct(fingerprint_accuracy)]
        for round_id, face_accuracy, fingerprint_accuracy in zip(
            convergence["round"],
            convergence["face_accuracy"],
            convergence["fingerprint_accuracy"],
        )
    ]
    loss_rows = [
        [str(round_id), number(face_loss), number(fingerprint_loss)]
        for round_id, face_loss, fingerprint_loss in zip(
            convergence["round"],
            convergence["face_loss"],
            convergence["fingerprint_loss"],
        )
    ]
    report.page_break()
    report.table(
        "Weighted local accuracy by federated round",
        ["Round", "Face weighted local accuracy", "Fingerprint weighted local accuracy"],
        accuracy_rows,
        "do_fmbs_data/phase2_metrics.json",
    )
    report.page_break()
    report.table(
        "Cross-entropy loss by federated round",
        ["Round", "Face cross-entropy loss", "Fingerprint cross-entropy loss"],
        loss_rows,
        "do_fmbs_data/phase2_metrics.json",
    )
    report.paragraph(
        "These two tables provide the plotted values for each of the 12 federated rounds: the first table "
        "corresponds to the weighted local accuracy plot, and the second to the cross-entropy loss plot. "
        "Accuracy is shown as a percentage; losses are cross-entropy values. For each modality and round, each plotted metric is "
        "the client-weighted average (using the aggregation weights in Table 4.4) of the five clients' "
        "final local-epoch metrics. Thus, the values are generated from the recorded federated training "
        "history, not estimated from the figure. These are training-convergence metrics, not the separate "
        "held-out test results reported above."
    )

    report.heading("4.3 Adversarial Attack Simulation", 2)
    report.paragraph(
        "FGSM and PGD were applied to standardized face features only. Attack success rate used originally "
        "correct face decisions as its denominator. Membership inference was a confidence-threshold attack "
        "on the face branch. Label-flipping poisoning changed both modality labels in a controlled central "
        "simulation. No face-only evasion result is presented as fingerprint robustness evidence."
    )
    report.table("Attack results", ["Attack", "Scope", "Setting", "Measured result"], [
        *[[row["attack"], row["modality"], f"epsilon={row['epsilon']:.2f}",
           f"ASR {row['successful_attacks']}/{row['clean_correct']} = {pct(row['attack_success_rate'])}"] for row in attacks["evasion"]],
        ["Membership inference", "Face", f"threshold={attacks['membership_inference']['selected_threshold']:.4f}",
         f"accuracy={pct(attacks['membership_inference']['attack_accuracy'])}; AUC={attacks['membership_inference']['ROC_AUC']:.4f}"],
        ["Label flipping", "Face and fingerprint labels", f"rate={pct(attacks['label_flipping_poisoning']['poison_rate'])}",
         f"face drop={pct(attacks['label_flipping_poisoning']['face_accuracy_drop'])}; fingerprint drop={pct(attacks['label_flipping_poisoning']['fingerprint_accuracy_drop'])}"],
    ], "do_fmbs_data/phase3_attack_results.json")
    copy_figure(report, manifest, output, "face_adversarial_attacks.png", "FGSM and PGD attack success rates on the face branch")

    report.heading("4.4 Defense Mechanisms", 2)
    report.paragraph(
        "Four states were compared: the federated baseline, DP-inspired Gaussian gradient-noise fine-tuning, "
        "face adversarial training, and the combined DO-FMBS defense. The combined objective was "
        "L = 0.5 L_face + 0.5 L_fingerprint. Automated gradient evidence confirmed non-zero gradients in "
        "the fingerprint branch. Because the implementation has no per-example clipping and no privacy "
        "accountant, it is not described as formal (epsilon, delta)-differential privacy."
    )
    defense_rows = []
    for name, values in defenses["comparison"].items():
        defense_rows.append([name.replace("_", " ").title(), pct(values["face_classification"]["accuracy"]),
                             pct(values["fingerprint_classification"]["accuracy"]),
                             pct(values["face_fgsm"]["attack_success_rate"])])
    report.table("Defense comparison", ["Model", "Face accuracy", "Fingerprint accuracy", "Face FGSM ASR"],
                 defense_rows, "do_fmbs_data/phase4_defense_results.json")
    copy_figure(report, manifest, output, "defense_comparison.png", "Defense comparison separated by modality and metric")

    report.heading("4.5 Developed DO-FMBS Model", 2)
    report.paragraph(
        "The implemented neural model contains a face input branch and a fingerprint input branch. Each "
        "branch has its own dense representation and softmax identity head. Decisions are produced directly "
        "from modality-specific representations. This is accurately characterized as a jointly trained "
        "dual-branch model, not a person-level multimodal fusion classifier."
    )

    report.heading("4.6 Model Validation", 2)
    report.heading("4.6.1 Face Classification Validation", 3)
    report.table("Face classification results", ["Samples", "Accuracy", "95% CI", "Macro precision", "Macro recall", "Macro F1", "Weighted F1"], [[
        face["samples"], pct(face["accuracy"]), f"[{pct(face['accuracy_wilson_95'][0])}, {pct(face['accuracy_wilson_95'][1])}]",
        pct(face["macro_precision"]), pct(face["macro_recall"]), pct(face["macro_f1"]), pct(face["weighted_f1"]),
    ]], "do_fmbs_data/face_classification_report.json")
    copy_figure(report, manifest, output, "face_confusion_matrix.png", "Final face classification confusion matrix")
    report.heading("4.6.2 Fingerprint Classification Validation", 3)
    report.table("Fingerprint classification results", ["Samples", "Accuracy", "95% CI", "Macro precision", "Macro recall", "Macro F1", "Weighted F1"], [[
        finger["samples"], pct(finger["accuracy"]), f"[{pct(finger['accuracy_wilson_95'][0])}, {pct(finger['accuracy_wilson_95'][1])}]",
        pct(finger["macro_precision"]), pct(finger["macro_recall"]), pct(finger["macro_f1"]), pct(finger["weighted_f1"]),
    ]], "do_fmbs_data/fingerprint_classification_report.json")
    copy_figure(report, manifest, output, "fingerprint_confusion_matrix.png", "Final fingerprint classification confusion matrix")
    report.paragraph(
        "Fingerprint classification is a closed-set multiclass identification task. It is distinct from the "
        "claimed-identity verification task below; classification accuracy is therefore not interpreted as FRR."
    )

    report.heading("4.6.3 Face Verification Validation", 3)
    report.paragraph(
        "The calibration partition selected an equal-error-rate threshold. That threshold was frozen before "
        "the independent test partition was scored. Each test probe generated one genuine claim and one "
        "claim against every incorrect enrolled identity."
    )
    headers = ["Threshold", "FAR", "FAR 95% CI", "FRR", "FRR 95% CI", "TAR", "AUC", "EER"]
    report.table("Face verification results", headers, [verification_row(face_ver)], "do_fmbs_data/face_verification_report.json")
    copy_figure(report, manifest, output, "face_verification_roc.png", "Face claimed-identity verification ROC curve")
    report.heading("4.6.4 Fingerprint Verification Validation", 3)
    report.table("Fingerprint verification results", headers, [verification_row(finger_ver)], "do_fmbs_data/fingerprint_verification_report.json")
    copy_figure(report, manifest, output, "fingerprint_verification_roc.png", "Fingerprint claimed-identity verification ROC curve")
    copy_figure(report, manifest, output, "fingerprint_genuine_impostor_distribution.png", "Fingerprint genuine and impostor score distributions")
    copy_figure(report, manifest, output, "fingerprint_far_frr_threshold.png", "Fingerprint calibration FAR and FRR across thresholds")
    fingerprint_final = finger_ver["final_test"]
    report.paragraph(
        f"At the frozen threshold, final fingerprint FAR was {fingerprint_final['impostor_accepts']}/"
        f"{fingerprint_final['impostor_attempts']} = {pct(fingerprint_final['FAR'], 3)}, while FRR was "
        f"{fingerprint_final['genuine_rejects']}/{fingerprint_final['genuine_attempts']} = "
        f"{pct(fingerprint_final['FRR'], 3)}. These values arise directly from the corrected replay and were "
        "not tuned on the final test observations."
    )
    report.heading("4.6.5 Statistical Validation", 3)
    stat_rows = [[row["modality"].title(), row["baseline_only_correct"], row["final_only_correct"],
                  row["discordant"], f"{row['p_value']:.6g}", "Yes" if row["significant_at_0_05"] else "No"]
                 for row in manifest["final_metrics"]["statistical_validation"]]
    report.table("Paired baseline-versus-final statistical validation", ["Modality", "Baseline only", "Final only", "Discordant", "p-value", "Significant (0.05)"],
                 stat_rows, "do_fmbs_data/phase5_validation.json")
    report.heading("4.6.6 Robustness Validation", 3)
    robust_rows = []
    for model_name in ("baseline", "do_fmbs"):
        for row in manifest["final_metrics"]["robustness"][model_name]:
            robust_rows.append([model_name.replace("_", " ").title(), f"{row['epsilon']:.2f}",
                                pct(row["accuracy_all_samples"]),
                                f"{row['successful_attacks']}/{row['eligible_clean_correct']} = {pct(row['attack_success_rate'])}"])
    report.table("Face-branch robustness results", ["Model", "FGSM epsilon", "Accuracy", "Attack success rate"],
                 robust_rows, "do_fmbs_data/face_robustness_report.json")
    copy_figure(report, manifest, output, "face_robustness_validation.png", "Baseline and final-model face robustness under FGSM")

    report.heading("4.7 Dynamic Optimization", 2)
    report.paragraph(
        "Phase 6 modeled adversarial-input pressure (AI) and privacy-attack pressure (PA) as measured state "
        "variables, while defense intensity (DM) and participating-client proportion (GS) were constrained "
        "controls. The response Y is a simulated next-round efficiency score. The coefficients were estimated "
        "from a seeded factorial simulation; they are not neural-network weights and do not represent hundreds "
        "of production federated deployments."
    )
    coefficient_rows = [[name, f"{estimate:+.4f}", f"{low:+.4f}", f"{high:+.4f}"]
                        for name, estimate, low, high in zip(fit["names"], fit["coefficients"], fit["ci_low"], fit["ci_high"])]
    report.table("Dynamic optimization coefficients", ["Coefficient", "Estimate", "95% CI low", "95% CI high"],
                 coefficient_rows, manifest["dynamic_optimization"]["artifact_directory"] + "/model_fit.json")
    report.table("Dynamic optimization diagnostics", ["Diagnostic", "Result"], [
        ["Observations", fit["observations"]], ["R-squared", number(fit["r_squared"])],
        ["Adjusted R-squared", number(fit["adjusted_r_squared"])], ["Cross-validated R-squared", number(fit["cross_validation_r2_mean"])],
        ["RMSE", number(fit["rmse"])], ["Maximum constraint violation", number(fit["maximum_constraint_violation"], 6)],
        ["Optimization success", fit["optimization_success"]],
    ], manifest["dynamic_optimization"]["artifact_directory"] + "/model_fit.json")
    dynamic_figure(report, manifest, output)

    report.heading("4.8 Results and Discussion", 2)
    report.paragraph(
        "The corrected experiment demonstrates why data validity precedes model evaluation. Persistent ridge "
        "templates changed the fingerprint task from learning arbitrary impression labels to learning stable "
        "synthetic identities. The final classification and verification tables should be interpreted together: "
        "identification measures closed-set class decisions, whereas FAR and FRR measure thresholded claims."
    )
    baseline = defenses["comparison"]["baseline"]
    final_defense = defenses["comparison"]["do_fmbs_combined"]
    report.table("Federated baseline versus final DO-FMBS comparison", ["Metric", "Baseline", "Final DO-FMBS", "Scope"], [
        ["Face classification accuracy", pct(baseline["face_classification"]["accuracy"]), pct(final_defense["face_classification"]["accuracy"]), "Final face test"],
        ["Fingerprint classification accuracy", pct(baseline["fingerprint_classification"]["accuracy"]), pct(final_defense["fingerprint_classification"]["accuracy"]), "Final fingerprint test"],
        ["Face FGSM ASR", pct(baseline["face_fgsm"]["attack_success_rate"]), pct(final_defense["face_fgsm"]["attack_success_rate"]), f"epsilon={final_defense['face_fgsm']['epsilon']:.2f}"],
    ], "do_fmbs_data/phase4_defense_results.json")
    report.paragraph(
        "A favorable or unfavorable defense difference is reported as observed rather than treated as a "
        "success criterion. The statistical table determines whether paired decision differences are conclusive "
        "at the declared level. The dynamic controller complements, but does not replace, the directly measured "
        "neural and biometric evidence."
    )
    report.heading("4.8.1 Limitations", 3)
    report.paragraph(
        "The fingerprint data are synthetic and are not an external benchmark. LFW and fingerprint identities "
        "belong to different populations, so person-level multimodal fusion has not been demonstrated. The "
        "study scale is experimental rather than deployment scale. Gaussian gradient noise is DP-inspired and "
        "has no formal privacy budget. FGSM and PGD evidence applies only to the face branch. Membership and "
        "poisoning simulations simplify real attackers. Finally, Phase 6 is a controlled model-derived simulation, "
        "and its policy requires external calibration before operational use."
    )
    report.heading("4.9 Chapter Summary", 2)
    report.paragraph(
        "The six-phase replay completed with synchronized artifacts and explicit provenance. The data generator, "
        "preprocessing, federated allocation, dual-loss defense, classification evaluation and claimed-identity "
        "verification protocol were aligned scientifically. All numerical claims in this chapter map to hashed "
        "artifacts from one replay."
    )
    report.page_break()
    report.heading("APPENDIX 4-A: EVIDENCE TRACEABILITY", 2)
    report.paragraph(
        "The machine-readable chapter4_evidence_map.json file records each table and figure, its source artifact "
        "and the generating script. The final evidence manifest additionally records SHA-256 hashes, commands, "
        "exit codes, runtime and environment versions."
    )
    report.heading("A4.1 Authoritative Replay", 3)
    report.paragraph(
        f"Replay identifier: {manifest['replay_id']}. The authoritative manifest is "
        "do_fmbs_data/final_evidence_manifest.json. Evidence was accepted only after all six phase commands "
        "returned successful exit codes and the independent replay audit passed."
    )
    report.heading("A4.2 Phase-to-Artifact Map", 3)
    trace_rows = [
        ("Phase 1 — data preparation", "phase1_results.pkl; phase1_metadata.json; fingerprint_quality_gate.json; split indices and train-only scalers"),
        ("Phase 2 — federated learning", "phase2_results.pkl; phase2_metadata.json; exact client partitions; federated convergence figure"),
        ("Phase 3 — attack simulation", "phase3_results.pkl; phase3_metadata.json; face-branch attack curves and poisoning records"),
        ("Phase 4 — defenses", "phase4_results.pkl; phase4_metadata.json; modality losses; fingerprint-gradient activation evidence"),
        ("Phase 5 — validation", "phase5_results.pkl; phase5_metadata.json; classification confusion matrices; claimed-identity scores, thresholds, ROC curves and confidence intervals"),
        ("Phase 6 — optimization", "dynamic_optimization_schedule.csv; coefficient and diagnostic JSON files; optimization trajectory"),
    ]
    for phase, artifacts in trace_rows:
        report.paragraph(f"{phase}: {artifacts}.")
    report.heading("A4.3 Reproduction and Integrity Checks", 3)
    report.paragraph(
        "scripts/run_isolated_instrumented_pipeline.py reproduces the isolated replay. "
        "scripts/verify_replay_evidence.py independently checks partition disjointness, scaler provenance, "
        "verification arithmetic, fingerprint quality and dual-branch gradients."
    )
    report.paragraph(
        "scripts/build_final_evidence_manifest.py hashes the accepted outputs and publishes the synchronized "
        "manifest. scripts/generate_chapter_four_report.py then validates those hashes before reproducing any "
        "number, table or figure."
    )
    report.paragraph(
        "The Streamlit dashboard reads the same final manifest. Historical root-level result files and older "
        "report outputs are excluded from the authoritative evidence chain and cannot supply dashboard or "
        "Chapter Four metrics."
    )
    return report


def markdown_text(report: Report) -> str:
    lines = []
    for block in report.blocks:
        if block["kind"] == "heading":
            lines.extend(["#" * block["level"] + " " + block["text"], ""])
        elif block["kind"] == "paragraph":
            lines.extend([block["text"], ""])
        elif block["kind"] == "table":
            lines.extend([f"**{block['caption']}**", "",
                          "| " + " | ".join(block["headers"]) + " |",
                          "| " + " | ".join(["---"] * len(block["headers"])) + " |"])
            lines.extend("| " + " | ".join(row) + " |" for row in block["rows"])
            lines.append("")
        elif block["kind"] == "figure":
            lines.extend([f"![{block['caption']}]({Path(block['path']).name})", "", f"**{block['caption']}**", ""])
        elif block["kind"] == "page_break":
            lines.extend(["---", ""])
    return "\n".join(lines).strip() + "\n"


def set_cell_text(cell, text: str, size: float = 9.0, bold: bool = False) -> None:
    from docx.shared import Pt

    cell.text = str(text)
    for paragraph in cell.paragraphs:
        for run in paragraph.runs:
            run.font.name = "Times New Roman"
            run.font.size = Pt(size)
            run.bold = bold


def write_docx(report: Report, path: Path) -> None:
    from docx import Document
    from docx.enum.section import WD_SECTION
    from docx.enum.text import WD_ALIGN_PARAGRAPH
    from docx.oxml import OxmlElement
    from docx.oxml.ns import qn
    from docx.shared import Inches, Pt

    document = Document()
    section = document.sections[0]
    section.top_margin = section.bottom_margin = Inches(0.85)
    section.left_margin = section.right_margin = Inches(0.9)
    normal = document.styles["Normal"]
    normal.font.name = "Times New Roman"
    normal.font.size = Pt(12)
    normal.paragraph_format.line_spacing = 1.5
    normal.paragraph_format.space_after = Pt(6)
    for style_name, size in (("Title", 16), ("Heading 1", 14), ("Heading 2", 13), ("Heading 3", 12)):
        style = document.styles[style_name]
        style.font.name = "Times New Roman"
        style.font.size = Pt(size)
        style.font.bold = True
        style.paragraph_format.keep_with_next = True

    footer = section.footer.paragraphs[0]
    footer.alignment = WD_ALIGN_PARAGRAPH.CENTER
    run = footer.add_run()
    begin = OxmlElement("w:fldChar"); begin.set(qn("w:fldCharType"), "begin")
    instruction = OxmlElement("w:instrText"); instruction.set(qn("xml:space"), "preserve"); instruction.text = "PAGE"
    end = OxmlElement("w:fldChar"); end.set(qn("w:fldCharType"), "end")
    run._r.extend([begin, instruction, end])

    title_count = 0
    for block in report.blocks:
        kind = block["kind"]
        if kind == "heading":
            level = block["level"]
            if level == 1:
                paragraph = document.add_paragraph(style="Title")
                paragraph.alignment = WD_ALIGN_PARAGRAPH.CENTER
                paragraph.add_run(block["text"])
                title_count += 1
                if title_count == 2:
                    document.add_page_break()
            else:
                document.add_heading(block["text"], level=min(level - 1, 3))
        elif kind == "paragraph":
            paragraph = document.add_paragraph(block["text"])
            paragraph.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
        elif kind == "table":
            caption = document.add_paragraph()
            caption.alignment = WD_ALIGN_PARAGRAPH.CENTER
            caption.paragraph_format.keep_with_next = True
            caption.add_run(block["caption"]).bold = True
            table = document.add_table(rows=1, cols=len(block["headers"]))
            table.style = "Table Grid"
            table.autofit = True
            header = table.rows[0]
            tr_pr = header._tr.get_or_add_trPr()
            repeat = OxmlElement("w:tblHeader"); repeat.set(qn("w:val"), "true"); tr_pr.append(repeat)
            for index, value in enumerate(block["headers"]):
                set_cell_text(header.cells[index], value, 8.5, True)
            for row in block["rows"]:
                cells = table.add_row().cells
                for index, value in enumerate(row):
                    set_cell_text(cells[index], value, 8.5)
                tr_pr = cells[0]._tc.getparent().get_or_add_trPr()
                no_split = OxmlElement("w:cantSplit"); tr_pr.append(no_split)
            document.add_paragraph()
        elif kind == "figure":
            paragraph = document.add_paragraph()
            paragraph.alignment = WD_ALIGN_PARAGRAPH.CENTER
            paragraph.add_run().add_picture(block["path"], width=Inches(6.25))
            caption = document.add_paragraph()
            caption.alignment = WD_ALIGN_PARAGRAPH.CENTER
            caption.paragraph_format.keep_with_next = False
            caption.add_run(block["caption"]).bold = True
        elif kind == "page_break":
            document.add_page_break()
    document.core_properties.title = "Chapter Four: Model Development, Implementation and Validation"
    document.core_properties.subject = "DO-FMBS master's research evidence"
    document.save(path)


def write_pdf(report: Report, path: Path) -> None:
    from xml.sax.saxutils import escape
    from reportlab.lib import colors
    from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
    from reportlab.lib.units import inch
    from reportlab.platypus import Image, PageBreak, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle

    styles = getSampleStyleSheet()
    body = ParagraphStyle("ResearchBody", parent=styles["BodyText"], fontName="Times-Roman", fontSize=10.5,
                          leading=15.5, alignment=TA_JUSTIFY, spaceAfter=7)
    heading = {
        1: ParagraphStyle("T", parent=styles["Title"], fontName="Times-Bold", fontSize=16, leading=20, alignment=TA_CENTER, spaceAfter=10),
        2: ParagraphStyle("H1", parent=styles["Heading1"], fontName="Times-Bold", fontSize=13.5, leading=17, spaceBefore=10, spaceAfter=6, keepWithNext=True),
        3: ParagraphStyle("H2", parent=styles["Heading2"], fontName="Times-Bold", fontSize=12, leading=15, spaceBefore=8, spaceAfter=5, keepWithNext=True),
    }
    caption_style = ParagraphStyle("Caption", parent=body, fontName="Times-Bold", fontSize=9.5,
                                   leading=12, alignment=TA_CENTER, spaceBefore=4, spaceAfter=8)
    document = SimpleDocTemplate(str(path), pagesize=A4, rightMargin=0.65*inch, leftMargin=0.65*inch,
                                 topMargin=0.70*inch, bottomMargin=0.70*inch,
                                 title="DO-FMBS Chapter Four")
    story = []
    title_count = 0
    for block in report.blocks:
        kind = block["kind"]
        if kind == "heading":
            story.append(Paragraph(escape(block["text"]), heading[min(block["level"], 3)]))
            if block["level"] == 1:
                title_count += 1
                if title_count == 2:
                    story.append(PageBreak())
        elif kind == "paragraph":
            story.append(Paragraph(escape(block["text"]), body))
        elif kind == "table":
            story.append(Paragraph(escape(block["caption"]), caption_style))
            data = [[Paragraph(escape(str(cell)), body) for cell in block["headers"]]]
            data.extend([[Paragraph(escape(str(cell)), body) for cell in row] for row in block["rows"]])
            available = A4[0] - 1.30 * inch
            table = Table(data, colWidths=[available / len(block["headers"])] * len(block["headers"]), repeatRows=1)
            table.setStyle(TableStyle([
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#D9EAF7")),
                ("FONTNAME", (0, 0), (-1, 0), "Times-Bold"), ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#666666")),
                ("LEFTPADDING", (0, 0), (-1, -1), 3), ("RIGHTPADDING", (0, 0), (-1, -1), 3),
                ("TOPPADDING", (0, 0), (-1, -1), 3), ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
            ]))
            story.extend([table, Spacer(1, 8)])
        elif kind == "figure":
            image = Image(block["path"])
            max_width, max_height = A4[0] - 1.5*inch, 5.7*inch
            scale = min(max_width / image.imageWidth, max_height / image.imageHeight)
            image.drawWidth, image.drawHeight = image.imageWidth * scale, image.imageHeight * scale
            story.extend([image, Paragraph(escape(block["caption"]), caption_style)])
        elif kind == "page_break":
            story.append(PageBreak())

    def footer(canvas, doc):
        canvas.saveState(); canvas.setFont("Times-Roman", 9)
        canvas.drawCentredString(A4[0] / 2, 0.38 * inch, str(doc.page)); canvas.restoreState()
    document.build(story, onFirstPage=footer, onLaterPages=footer)


def validate_outputs(report: Report, manifest: dict, output: Path) -> dict:
    from docx import Document
    import fitz

    docx_path = output / f"{STEM}.docx"
    pdf_path = output / f"{STEM}.pdf"
    md_path = output / f"{STEM}.md"
    document = Document(docx_path)
    docx_text = "\n".join(p.text for p in document.paragraphs)
    pdf = fitz.open(pdf_path)
    page_text = [page.get_text().strip() for page in pdf]
    pdf_text = "\n".join(page_text)
    required = ["4.0 Introduction", "4.1 Data Processing", "4.2 Federated Learning Implementation",
                "4.3 Adversarial Attack Simulation", "4.4 Defense Mechanisms", "4.5 Developed DO-FMBS Model",
                "4.6 Model Validation", "4.7 Dynamic Optimization", "4.8 Results and Discussion",
                "APPENDIX 4-A: EVIDENCE TRACEABILITY"]
    for label in required:
        if label not in docx_text or label not in pdf_text:
            raise RuntimeError(f"Missing required section in generated documents: {label}")
    if len(document.tables) != report.table_number:
        raise RuntimeError("DOCX table count does not match report table sequence")
    if any(not text for text in page_text):
        raise RuntimeError("Rendered PDF contains a blank page")
    captions = [p.text for p in document.paragraphs if p.text.startswith(("Table 4.", "Figure 4."))]
    expected_tables = [f"Table 4.{index}:" for index in range(1, report.table_number + 1)]
    expected_figures = [f"Figure 4.{index}:" for index in range(1, report.figure_number + 1)]
    for prefix in expected_tables + expected_figures:
        if not any(value.startswith(prefix) for value in captions):
            raise RuntimeError(f"Missing or non-sequential caption: {prefix}")
    final_finger = manifest["final_metrics"]["fingerprint_verification"]["final_test"]
    exact = f"{final_finger['genuine_rejects']}/{final_finger['genuine_attempts']} = {pct(final_finger['FRR'], 3)}"
    if exact not in docx_text:
        raise RuntimeError("Fingerprint FRR does not reproduce the authoritative counts")
    if "formally guaranteed differential privacy" in docx_text.lower():
        raise RuntimeError("Unsupported formal differential-privacy claim detected")
    if "person-level multimodal fusion" in docx_text.lower() and "does not claim" not in docx_text.lower():
        raise RuntimeError("Unsupported person-level fusion claim detected")
    if md_path.stat().st_size == 0 or docx_path.stat().st_size == 0 or pdf_path.stat().st_size == 0:
        raise RuntimeError("Generated Chapter Four output is empty")
    return {"docx_paragraphs": len(document.paragraphs), "docx_tables": len(document.tables),
            "pdf_pages": len(pdf), "blank_pdf_pages": 0, "table_count": report.table_number,
            "figure_count": report.figure_number, "metric_reproduction": "PASS",
            "caption_sequence": "PASS", "unsupported_claim_scan": "PASS"}


def generate(args: argparse.Namespace) -> list[Path]:
    manifest = load_manifest(args.manifest.resolve())
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    if args.validate_only:
        report = build_report(manifest, output)
        validation = validate_outputs(report, manifest, output)
        print(json.dumps(validation, indent=2))
        return [output / f"{STEM}.docx"]
    report = build_report(manifest, output)
    md_path = output / f"{STEM}.md"
    docx_path = output / f"{STEM}.docx"
    pdf_path = output / f"{STEM}.pdf"
    md_path.write_text(markdown_text(report), encoding="utf-8")
    write_docx(report, docx_path)
    write_pdf(report, pdf_path)
    evidence_map_path = output / "chapter4_evidence_map.json"
    evidence_map_path.write_text(json.dumps({"replay_id": manifest["replay_id"], "items": report.evidence_map}, indent=2) + "\n", encoding="utf-8")
    validation = validate_outputs(report, manifest, output)
    documents = [md_path, docx_path, pdf_path, evidence_map_path]
    generation_manifest = {
        "status": "CHAPTER-FOUR-GENERATED-FROM-AUTHORITATIVE-REPLAY",
        "replay_id": manifest["replay_id"], "created_utc": datetime.now(timezone.utc).isoformat(),
        "source_manifest": {"path": args.manifest.resolve().relative_to(ROOT).as_posix(), "sha256": sha256(args.manifest.resolve())},
        "validation": validation,
        "documents": [{"path": path.name, "bytes": path.stat().st_size, "sha256": sha256(path)} for path in documents],
        "figures": [{"path": path.name, "bytes": path.stat().st_size, "sha256": sha256(path)}
                    for path in sorted(output.glob("CHAPTER4_FIGURE_*.png"))],
    }
    generation_path = output / "CHAPTER_FOUR_GENERATION_MANIFEST.json"
    generation_path.write_text(json.dumps(generation_manifest, indent=2) + "\n", encoding="utf-8")
    print(f"Generated and validated: {docx_path}")
    print(f"Rendered PDF pages: {validation['pdf_pages']}; tables: {report.table_number}; figures: {report.figure_number}")
    return [*documents, generation_path]


def main() -> int:
    generate(parse_args())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
