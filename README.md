# DO-FMBS: Dynamic Optimization for Federated Multimodal Biometric Security

Master's research implementation for Jaramogi Oginga Odinga University of Science and Technology.

Researcher: Lydiah Moraa Machora (I152/P/9515/23)  
Supervisors: Dr. Richard Omollo, PhD; Dr. Gonza Omoro, PhD  
Year: 2026

## Scientific scope

DO-FMBS is an experimental, six-phase federated biometric pipeline with two independent identity spaces:

- face classification uses the public Labeled Faces in the Wild subset supplied by scikit-learn;
- fingerprint classification uses generated identity-persistent ridge templates with eight impressions per identity;
- the branches are trained together for computational convenience but are not person-level multimodal pairs;
- evasion and robustness experiments apply to the face branch only;
- Gaussian gradient noise is DP-inspired regularization, not formal differential privacy and carries no epsilon/delta guarantee;
- Phase 6 is a controlled dynamic-optimization simulation, not evidence from a production deployment.

The corrected fingerprint generator creates one stable ridge template per synthetic identity and derives impressions through bounded rotation/translation, elastic deformation, pressure variation, blur, sensor noise and partial contact. Phase 1 refuses to continue unless the same-identity/different-identity quality gate passes.

## Authoritative replay

The accepted replay is `final_scientific_replay_20260907`. Its only authoritative numerical index is:

`do_fmbs_data/final_evidence_manifest.json`

That manifest records the replay identifier, parameters, seeds, environment, phase runtimes, independent audit result, final metrics, scope statements and SHA-256 hashes for 68 evidence artifacts. Chapter Four and the Streamlit application validate and read this manifest; they do not read legacy root-level result files.

Headline final-test results are:

| Task | Face | Fingerprint |
|---|---:|---:|
| Closed-set classification accuracy | 272/454 = 59.91% | 191/200 = 95.50% |
| Claimed-identity FAR | 3129/27694 = 11.298% | 311/19800 = 1.571% |
| Claimed-identity FRR | 46/454 = 10.132% | 5/200 = 2.500% |
| Claimed-identity TAR | 89.868% | 97.500% |
| Verification ROC AUC | 0.9589 | 0.9948 |

Verification thresholds were selected at the equal-error operating point on calibration data and frozen before test evaluation. Every test sample produces one genuine claim and one claim against every incorrect enrolled identity. Classification accuracy and verification FAR/FRR are intentionally reported as different tasks.

## Pipeline

| Phase | Implementation | Main evidence |
|---|---|---|
| 1. Data preparation | `phase1_data_preparation.py` | split indices, train-only scalers, persistent fingerprints, quality gate |
| 2. Federated learning | `phase2_federated_learning.py` | exact five-client class-exclusive non-IID partitions, FedAvg history |
| 3. Attack simulation | `phase3_adversarial_attacks.py` | face-only FGSM/PGD and membership evidence; poisoning simulation |
| 4. Defenses | `phase4_defense_mechanisms.py` | baseline, DP-inspired, ART and combined dual-loss configurations |
| 5. Validation | `phase5_model_validation.py` | per-modality classification, verification, confidence intervals and robustness |
| 6. Dynamic optimization | `phase6_constrained_dynamic_optimization.py` | constrained coefficients, diagnostics and round-specific control schedule |

Both heads contribute to Phase 4 optimization:

```text
combined_loss = 0.5 * face_loss + 0.5 * fingerprint_loss
```

The replay also records nonzero fingerprint-head gradient norms so accidental removal of the fingerprint loss fails validation.

## Reproduce and verify

Activate the repository virtual environment first. On PowerShell:

```powershell
.\venv\Scripts\Activate.ps1
python scripts\run_isolated_instrumented_pipeline.py
python scripts\verify_replay_evidence.py --run-dir runtime_recording\final_scientific_replay_20260907
python scripts\build_final_evidence_manifest.py --run-dir runtime_recording\final_scientific_replay_20260907
python -m unittest discover -s tests -v
```

The isolated runner copies the phase code into a replay directory and writes phase-specific logs, commands, exit codes and timings. It does not mix those results with older root-level artifacts.

## Dashboard

Start the manifest-backed dashboard with:

```powershell
streamlit run app.py
```

The dashboard now provides an Overview plus a sidebar for Phases 1–6 and Evidence & Provenance. Each phase explains its purpose, prerequisites, inputs and outputs, shows the recorded configuration and results, and runs in order with streamed console output. A new simulation writes into its own replay folder; after Phase 6 the dashboard runs the independent audit and builds that replay's manifest. The existing accepted manifest remains available as read-only evidence while a new replay is in progress.

## Chapter Four

Generate and validate the final chapter with:

```powershell
python scripts\generate_chapter_four_report.py
```

Outputs:

- `docs/chapter_four/CHAPTER_FOUR_MODEL_DEVELOPMENT_IMPLEMENTATION_AND_VALIDATION.docx`
- `docs/chapter_four/CHAPTER_FOUR_MODEL_DEVELOPMENT_IMPLEMENTATION_AND_VALIDATION.pdf`
- `docs/chapter_four/CHAPTER_FOUR_MODEL_DEVELOPMENT_IMPLEMENTATION_AND_VALIDATION.md`
- `docs/chapter_four/chapter4_evidence_map.json`
- `docs/chapter_four/CHAPTER_FOUR_GENERATION_MANIFEST.json`

The generator verifies every source hash, section presence, caption sequence, reproduced metric, table/figure count and rendered-page text before declaring success.

## Dynamic optimization

Phase 6 fits the constrained round-ahead model

```text
Y_(t+1) = B0 + B_AI*AI_t + B_PA*PA_t + B_DM*DM_t + B_GS*GS_t
```

where AI and PA are measured threat states, while DM and GS are optimized defense and client-participation controls. In the accepted seeded simulation, R-squared is 0.9681, adjusted R-squared is 0.9679 and cross-validated R-squared is 0.9673. All declared coefficient constraints pass. These diagnostics validate the controlled simulation only.

## Historical artifacts

Older result pickles, figures, reports and ancillary evaluators remain in the repository to preserve prior work. They are non-authoritative and may reflect superseded protocols, including per-impression fingerprint labels or random verification pairing. Do not quote them as final findings. The authoritative manifest, replay audit, current dashboard and current Chapter Four are the synchronized evidence chain.

## Requirements

- Python 3.10 recommended
- 8 GB RAM minimum; 16 GB recommended
- CPU execution is supported; a compatible GPU is optional
- install dependencies with `pip install -r requirements_app.txt`
