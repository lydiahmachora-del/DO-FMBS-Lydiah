"""Independent verification for a persisted authoritative Phase 5 run."""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import numpy as np


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def check(condition: bool, name: str, detail: str, checks: list[dict]) -> None:
    checks.append({"check": name, "status": "PASS" if condition else "FAIL", "detail": detail})
    if not condition:
        raise AssertionError(f"{name}: {detail}")


def _classification(labels: np.ndarray, probabilities: np.ndarray, class_count: int) -> tuple[dict, np.ndarray]:
    from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
    prediction = probabilities.argmax(axis=1)
    report = classification_report(labels, prediction, labels=np.arange(class_count), output_dict=True, zero_division=0)
    return {"correct": int(np.count_nonzero(prediction == labels)),
            "confusion_matrix": confusion_matrix(labels, prediction, labels=np.arange(class_count)).tolist(),
            "macro_precision": float(report["macro avg"]["precision"]),
            "macro_recall": float(report["macro avg"]["recall"]),
            "macro_f1": float(report["macro avg"]["f1-score"])}, prediction


def _verification(scores: np.ndarray, genuine: np.ndarray, threshold: float) -> dict:
    from sklearn.metrics import roc_auc_score, roc_curve
    accepted = scores >= threshold
    fpr, tpr, thresholds = roc_curve(genuine.astype(np.int8), scores)
    index = int(np.argmin(np.abs(fpr - (1.0 - tpr))))
    genuine_n, impostor_n = int(genuine.sum()), int((~genuine).sum())
    genuine_accepts = int(np.count_nonzero(accepted & genuine))
    impostor_accepts = int(np.count_nonzero(accepted & ~genuine))
    return {"genuine_attempts": genuine_n, "impostor_attempts": impostor_n,
            "genuine_accepts": genuine_accepts, "genuine_rejects": genuine_n - genuine_accepts,
            "impostor_accepts": impostor_accepts, "impostor_rejects": impostor_n - impostor_accepts,
            "FAR": impostor_accepts / impostor_n, "FRR": (genuine_n - genuine_accepts) / genuine_n,
            "TAR": genuine_accepts / genuine_n, "ROC_AUC": float(roc_auc_score(genuine, scores)),
            "EER": float((fpr[index] + 1.0 - tpr[index]) / 2.0),
            "EER_threshold": float(thresholds[index])}


def _paired_statistics(labels: np.ndarray, baseline: np.ndarray, candidate: np.ndarray,
                       seed: int, permutations: int, bootstraps: int) -> dict:
    from scipy.stats import binomtest
    identities = np.unique(labels)
    cluster_delta = np.asarray([np.count_nonzero((candidate == labels) & (labels == value))
                                - np.count_nonzero((baseline == labels) & (labels == value))
                                for value in identities], dtype=np.int64)
    observed = int(cluster_delta.sum())
    rng = np.random.default_rng(seed)
    randomized = rng.choice(np.asarray([-1, 1], dtype=np.int8), size=(permutations, len(cluster_delta))) @ cluster_delta
    pvalue = (1 + np.count_nonzero(np.abs(randomized) >= abs(observed))) / (permutations + 1)
    clusters = [np.flatnonzero(labels == value) for value in identities]
    rng = np.random.default_rng(seed + 1)
    differences = np.empty(bootstraps)
    for i in range(bootstraps):
        sampled = np.concatenate([clusters[j] for j in rng.integers(0, len(identities), len(identities))])
        differences[i] = (np.count_nonzero(candidate[sampled] == labels[sampled])
                          - np.count_nonzero(baseline[sampled] == labels[sampled])) / len(sampled)
    base_ok, candidate_ok = baseline == labels, candidate == labels
    base_only = int(np.count_nonzero(base_ok & ~candidate_ok))
    candidate_only = int(np.count_nonzero(candidate_ok & ~base_ok))
    discordant = base_only + candidate_only
    mcnemar = 1.0 if not discordant else float(binomtest(min(base_only, candidate_only), discordant).pvalue)
    return {"p_value": float(pvalue), "paired_accuracy_difference": float(observed / len(labels)),
            "paired_cluster_bootstrap_95": np.quantile(differences, [.025, .975]).tolist(),
            "significant": bool(pvalue < .05), "baseline_only_correct": base_only,
            "candidate_only_correct": candidate_only, "mcnemar_record_level_sensitivity_p": mcnemar,
            "significance_level": .05}


def verify(run_dir: Path) -> dict:
    import tensorflow as tf
    from do_fmbs_core import modality_probabilities, select_eer_threshold
    from phase3_authoritative_attacks import fgsm
    DOC_SHA256 = "E3D90110AC8D4B81C7D70246FA02D3C510352DC064AF6C44792DE35E44577F5A"
    P1_RUN = "phase1_final_candidate_20261123_182623"
    P2_RUN = "phase2_authoritative_20261123_203617_remediation"
    P3_RUN = "phase3_authoritative_20261123_220738"
    P4_RUN = "phase4_authoritative_20261123_224549"
    epsilons = (0.0, .05, .10, .15, .20)
    permutations, bootstraps = 20000, 2000

    run_dir = run_dir.resolve()
    root = run_dir.parents[1]
    checks: list[dict] = []
    config = json.loads((run_dir / "phase5_config.json").read_text(encoding="utf-8"))
    metrics = json.loads((run_dir / "phase5_metrics.json").read_text(encoding="utf-8"))
    manifest = json.loads((run_dir / "phase5_manifest.json").read_text(encoding="utf-8"))
    check(config["run_id"] == run_dir.name == metrics["run_id"] == manifest["run_id"],
          "Phase 5 run identity", run_dir.name, checks)

    bad = []
    for item in manifest.get("artifacts", []):
        target = run_dir / item["path"]
        if not target.is_file() or target.stat().st_size != item["bytes"] or sha256(target) != item["sha256"]:
            bad.append(item["path"])
    check(not bad and len(manifest.get("artifacts", [])) >= 10, "Phase 5 manifest artifacts",
          f"listed={len(manifest.get('artifacts', []))}; bad={bad}", checks)

    doc = root / "docs/reference/CHAPTER_FOUR_MODEL_DEVELOPMENT_IMPLEMENTATION_AND_VALIDATION.docx"
    check(sha256(doc).upper() == DOC_SHA256 == config["source_document_sha256"],
          "Authoritative Chapter Four identity", DOC_SHA256, checks)
    parent_specs = ((P1_RUN, "phase1_manifest.json", "source_phase1_manifest_sha256"),
                    (P2_RUN, "phase2_manifest.json", "source_phase2_manifest_sha256"),
                    (P3_RUN, "phase3_manifest.json", "source_phase3_manifest_sha256"),
                    (P4_RUN, "phase4_manifest.json", "source_phase4_manifest_sha256"))
    for parent, manifest_name, field in parent_specs:
        path = root / "runtime_recording" / parent / manifest_name
        check(path.is_file() and sha256(path) == config[field], f"Parent evidence {parent}",
              f"manifest_sha256={config[field]}", checks)
    check(config["source_phase1_run"] == P1_RUN and config["source_phase2_run"] == P2_RUN
          and config["source_phase3_run"] == P3_RUN and config["source_phase4_run"] == P4_RUN,
          "Accepted source run identities", "P1/P2/P3/P4 identifiers match", checks)

    with np.load(run_dir / "evaluation_records.npz", allow_pickle=False) as z:
        records = {key: z[key].copy() for key in z.files}
    p1 = root / "runtime_recording" / P1_RUN
    p4 = root / "runtime_recording" / P4_RUN
    with np.load(p1 / "face_data.npz", allow_pickle=False) as f, np.load(p1 / "fingerprint_data.npz", allow_pickle=False) as g:
        check(np.array_equal(records["face_eval_x"], f["X_test"][:100])
              and np.array_equal(records["face_eval_y"], f["y_test"][:100])
              and np.array_equal(records["finger_eval_x"], g["X_test"][:100])
              and np.array_equal(records["finger_eval_y"], g["y_test"][:100]),
              "Paired first-100 evaluation rows", "Both modalities match the accepted Phase 1 test arrays", checks)
        face_classes, finger_classes = int(f["n_classes"][0]), int(g["n_classes"][0])
        full = {"face_x": f["X_test"].astype(np.float32), "face_y": f["y_test"].astype(np.int64),
                "face_ids": f["test_indices"].astype(np.int64), "finger_x": g["X_test"].astype(np.float32),
                "finger_y": g["y_test"].astype(np.int64), "finger_ids": g["test_indices"].astype(np.int64)}
    with np.load(p4 / "evaluation_records.npz", allow_pickle=False) as z:
        check(np.array_equal(records["face_eval_ids"], z["face_eval_ids"])
              and np.array_equal(records["finger_eval_ids"], z["finger_eval_ids"]),
              "Phase 4 evaluation record linkage", "Saved record IDs equal Phase 4 accepted evaluation", checks)

    with np.load(run_dir / "verification_claim_records.npz", allow_pickle=False) as z:
        claim_data = {key: z[key].copy() for key in z.files}
    with np.load(p1 / "verification/face_claims.npz", allow_pickle=False) as z:
        check(all(np.array_equal(claim_data[f"face_{k}"], z[k]) for k in z.files),
              "Face claim provenance", "Saved Phase 1 claims retained without change", checks)
    with np.load(p1 / "verification/fingerprint_claims.npz", allow_pickle=False) as z:
        check(all(np.array_equal(claim_data[f"fingerprint_{k}"], z[k]) for k in z.files),
              "Fingerprint claim provenance", "Saved Phase 1 claims retained without change", checks)

    models = {
        "baseline": tf.keras.models.load_model(run_dir / "baseline_model.keras", compile=False),
        "dofmbs_combined": tf.keras.models.load_model(run_dir / "combined_dofmbs_model.keras", compile=False),
    }
    evidence_file = np.load(run_dir / "classification_and_verification_evidence.npz", allow_pickle=False)
    attack_file = np.load(run_dir / "robustness_evidence.npz", allow_pickle=False)
    for name, model in models.items():
        fp, gp = modality_probabilities(model, "face", records["face_eval_x"]), modality_probabilities(model, "fingerprint", records["finger_eval_x"])
        for modality, probs, labels, classes in (("face", fp, records["face_eval_y"], face_classes),
                                                  ("fingerprint", gp, records["finger_eval_y"], finger_classes)):
            summary, pred = _classification(labels, probs, classes)
            saved_prob = evidence_file[f"{name}_{modality}_probabilities"]
            saved_pred = evidence_file[f"{name}_{modality}_predictions"]
            saved_summary = metrics["models"][name]["face_classification" if modality == "face" else "fingerprint_classification"]
            check(np.allclose(probs, saved_prob, rtol=1e-5, atol=1e-6) and np.array_equal(pred, saved_pred)
                  and summary["correct"] == saved_summary["correct"]
                  and np.array_equal(summary["confusion_matrix"], saved_summary["confusion_matrix"])
                  and all(np.isclose(summary[field], saved_summary[field]) for field in ("macro_precision", "macro_recall", "macro_f1")),
                  f"{name} {modality} outputs and metrics", f"records={len(labels)} correct={summary['correct']}", checks)

        for modality, x, y, ids, classes in (("face", full["face_x"], full["face_y"], full["face_ids"], face_classes),
                                              ("fingerprint", full["finger_x"], full["finger_y"], full["finger_ids"], finger_classes)):
            prob = modality_probabilities(model, modality, x)
            prefix = "face" if modality == "face" else "fingerprint"
            cal_pos = {int(v): i for i, v in enumerate(ids)}
            cal_probe = claim_data[f"{prefix}_calibration_claim_probe_indices"]
            eval_probe = claim_data[f"{prefix}_evaluation_claim_probe_indices"]
            cal_rows = np.asarray([cal_pos[int(v)] for v in cal_probe])
            eval_rows = np.asarray([cal_pos[int(v)] for v in eval_probe])
            cal_ids = claim_data[f"{prefix}_calibration_claim_identity"].astype(int)
            eval_claim_ids = claim_data[f"{prefix}_evaluation_claim_identity"].astype(int)
            cal_mask = claim_data[f"{prefix}_calibration_claim_is_genuine"].astype(bool)
            eval_mask = claim_data[f"{prefix}_evaluation_claim_is_genuine"].astype(bool)
            genuine_cal = prob[cal_rows, cal_ids][cal_mask]
            impostor_cal = prob[cal_rows, cal_ids][~cal_mask]
            candidates = np.unique(np.concatenate(([0.0, 1.0], genuine_cal, impostor_cal)))
            if len(candidates) > 2500:
                candidates = np.quantile(candidates, np.linspace(0.0, 1.0, 2500))
            threshold = min(candidates, key=lambda t: (abs(np.mean(impostor_cal >= t) - np.mean(genuine_cal < t)),
                                                        np.mean(impostor_cal >= t) + np.mean(genuine_cal < t)))
            scores = prob[eval_rows, eval_claim_ids]
            expected = _verification(scores, eval_mask, float(threshold))
            saved = metrics["models"][name][f"{modality}_verification"]
            check(np.isclose(threshold, saved["threshold"]) and expected["genuine_attempts"] == saved["evaluation"]["genuine_attempts"]
                  and expected["impostor_attempts"] == saved["evaluation"]["impostor_attempts"]
                  and np.isclose(expected["FAR"], saved["evaluation"]["FAR"])
                  and np.isclose(expected["FRR"], saved["evaluation"]["FRR"])
                  and np.isclose(expected["TAR"], saved["evaluation"]["TAR"])
                  and np.isclose(expected["ROC_AUC"], saved["evaluation"]["ROC_AUC"])
                  and np.isclose(expected["EER"], saved["evaluation"]["EER"]),
                  f"{name} {modality} frozen-threshold verification", f"threshold={threshold:.8g}; claims={len(scores)}", checks)

        clean = evidence_file[f"{name}_face_predictions"]
        clean_mask = clean == records["face_eval_y"]
        expected_attack = {}
        robust_rows = []
        for epsilon in epsilons:
            adversarial = (records["face_eval_x"].copy() if epsilon == 0 else
                           fgsm(model, records["face_eval_x"], records["finger_eval_x"], records["face_eval_y"], epsilon))
            attacked_probability = modality_probabilities(model, "face", adversarial)
            attacked_prediction = attacked_probability.argmax(axis=1)
            successes = int(np.count_nonzero(clean_mask & (attacked_prediction != records["face_eval_y"])))
            robust_rows.append({"epsilon": epsilon, "clean_correct_eligible": int(clean_mask.sum()),
                                "attack_successes": successes, "attack_success_rate": successes / int(clean_mask.sum())})
            expected_attack[f"{name}_eps_{epsilon:.2f}_adversarial_face"] = adversarial.astype(np.float32)
        saved_rows = metrics["robustness"]["models"][name]
        for got, want in zip(saved_rows, robust_rows):
            check(got["epsilon"] == want["epsilon"] and got["clean_correct_eligible"] == want["clean_correct_eligible"]
                  and got["attack_successes"] == want["attack_successes"]
                  and np.isclose(got["attack_success_rate"], want["attack_success_rate"]),
                  f"{name} FGSM epsilon {got['epsilon']:.2f}", f"ASR={got['attack_success_rate']:.6f}", checks)
            saved_adv = attack_file[f"{name}_eps_{got['epsilon']:.2f}_adversarial_face"]
            key = f"{name}_eps_{got['epsilon']:.2f}_adversarial_face"
            check(np.array_equal(saved_adv, expected_attack[key]), f"{name} epsilon {got['epsilon']:.2f} attack inputs",
                  "Persisted attack input is reproducible", checks)

    labels = {"face": records["face_eval_y"], "fingerprint": records["finger_eval_y"]}
    for modality in labels:
        suffix = "face" if modality == "face" else "fingerprint"
        result = _paired_statistics(labels[modality], evidence_file[f"baseline_{suffix}_predictions"],
            evidence_file[f"dofmbs_combined_{suffix}_predictions"], config["seeds"][f"{modality}_sign_flip"], permutations, bootstraps)
        saved = metrics["statistical_comparisons"][modality]
        check(np.isclose(result["p_value"], saved["p_value"]) and np.isclose(result["paired_accuracy_difference"], saved["paired_accuracy_difference"])
              and np.allclose(result["paired_cluster_bootstrap_95"], saved["paired_cluster_bootstrap_95"])
              and result["baseline_only_correct"] == saved["baseline_only_correct"]
              and result["candidate_only_correct"] == saved["candidate_only_correct"]
              and np.isclose(result["mcnemar_record_level_sensitivity_p"], saved["mcnemar_record_level_sensitivity_p"])
              and result["significant"] == saved["significant"], f"{modality} paired statistical test",
              f"p={result['p_value']:.6g}; alpha={result['significance_level']}", checks)

    # Source snapshots are content-addressed in the run configuration.
    snap_checks = {name: (run_dir / "source_snapshot" / name).is_file()
                   and sha256(run_dir / "source_snapshot" / name) == digest
                   for name, digest in config["source_snapshot_sha256"].items()}
    check(all(snap_checks.values()), "Source snapshot hashes", json.dumps(snap_checks, sort_keys=True), checks)
    report = {"phase": 5, "run_id": run_dir.name, "status": "PASS", "checks": checks,
              "check_count": len(checks), "failed_checks": [], "independent": True}
    return report


if __name__ == "__main__":
    result = verify(Path(sys.argv[1]))
    print(json.dumps(result, sort_keys=True))
