# DO-FMBS: Privacy-Preserving Dynamic Optimization for Federated Multimodal Biometrics

Master's thesis implementation for Jaramogi Oginga Odinga University of Science and Technology.

**Researcher:** Lydiah Moraa Machora (I152/P/9515/23)

**Supervisors:** Dr. Richard Omollo, PhD; Dr. Gonza Omoro, PhD

**Year:** 2026

## Project structure

DO-FMBS combines six reproducible phases:

| Phase | Function | Terminal entry point |
|---|---|---|
| 1 | Prepare LFW face data and generated fingerprint data | `python phase1_data_preparation.py` |
| 2 | Train the federated multimodal baseline | `python phase2_federated_learning.py` |
| 3 | Apply evasion, membership-inference, and poisoning attacks | `python phase3_adversarial_attacks.py` |
| 4 | Train and compare defense configurations | `python phase4_defense_mechanisms.py` |
| 5 | Validate classification and robustness results | `python phase5_model_validation.py` |
| 6 | Estimate the constrained dynamic equation and optimize each round | `python phase6_constrained_dynamic_optimization.py` |

The interactive application is started with:

```powershell
streamlit run app.py
```

## Current experimental model comparison

The table below is read from `do_fmbs_data/final_results.pkl`, generated on 8 August 2026. “Privacy score” is the experiment's attack-resistance score, `1 − ASR`; it is not a formal differential-privacy guarantee.

| Model | Face accuracy | Attack success rate | Attack-resistance score |
|---|---:|---:|---:|
| Baseline | 57.19% | 42.86% | 57.14% |
| Differential privacy/noise | 51.07% | 52.38% | 47.62% |
| Adversarial training | 59.67% | 53.85% | 46.15% |
| DO-FMBS combined defense | 58.84% | 58.62% | 41.38% |

These figures describe this saved run and should be interpreted together with the Phase 3–5 methods and sample sizes. They are not used as substitutes for regression coefficients.

## Constrained dynamic model

Phase 6 fits the round-ahead equation:

$$
\widehat{Y}_{t+1}=B_0+B_{AI}AI_t+B_{PA}PA_t+B_{DM}DM_t+B_{GS}GS_t.
$$

The variables have one fixed operational meaning throughout estimation and optimization:

| Variable | Definition | Role | Domain |
|---|---|---|---:|
| $AI_t$ | Adversarial-input pressure, normalized as `epsilon / 0.30` | Measured threat state | 0–1 |
| $PA_t$ | `2 × max(0, membership-attack accuracy − 0.50)` | Measured privacy-threat state | fitted: 0–0.50 |
| $DM_t$ | Continuous defense intensity | Optimized control | 0–1 |
| $GS_t$ | `(participating clients − 1) / 4` | Optimized control | 0–1 |
| $Y_{t+1}$ | Simulated next-round FMBS efficiency score | Response | 0–1 |

“Dynamic” has a precise meaning here: the system measures $AI_t$ and $PA_t$ before a federated round, chooses $DM_t$ and $GS_t$, predicts $Y_{t+1}$, and repeats the optimization when the threat state changes in the next round.

### Estimated coefficients

The reproducible seed-42 design contains 700 balanced observations (`7 AI × 5 PA × 4 DM × 5 GS`). Its declared simulation ground truth is `(0.40, −0.18, −0.22, +0.26, +0.16)`, with seeded random error added before fitting. Sign-constrained least squares then recovers the values below. It enforces $B_{AI}\leq0$, $B_{PA}\leq0$, $B_{DM}\geq0$, and $B_{GS}\geq0$ during estimation; the ordering $B_{PA}\leq B_{AI}$ is evaluated separately rather than forced.

| Parameter | Estimate | Standard error | 95% confidence interval | Interpretation for a +1.0 change |
|---|---:|---:|---:|---|
| $B_0$ | 0.4007 | 0.0025 | [0.3959, 0.4056] | Reference performance when all normalized predictors are zero |
| $B_{AI}$ | −0.1821 | 0.0028 | [−0.1876, −0.1767] | Higher adversarial pressure lowers predicted performance |
| $B_{PA}$ | −0.2273 | 0.0050 | [−0.2372, −0.2175] | Higher privacy-attack pressure lowers predicted performance |
| $B_{DM}$ | +0.2614 | 0.0025 | [0.2565, 0.2663] | Stronger defense raises predicted performance |
| $B_{GS}$ | +0.1594 | 0.0026 | [0.1543, 0.1646] | Greater client participation raises predicted performance |

The fitted equation is therefore:

$$
\widehat{Y}_{t+1}=0.4007-0.1821AI_t-0.2273PA_t+0.2614DM_t+0.1594GS_t.
$$

### Fit and constraint evidence

| Check | Result | Acceptance gate | Decision |
|---|---:|---:|---|
| $R^2$ | 0.9681 | ≥ 0.90 | Pass |
| Adjusted $R^2$ | 0.9679 | ≥ 0.90 | Pass |
| Five-fold constrained cross-validation $R^2$ | 0.9673 | ≥ 0.89 | Pass |
| RMSE | 0.0245 | ≤ 0.06 | Pass |
| Maximum VIF | 1.0000 | < 2.0 | Pass |
| Maximum fitted-constraint violation | 0.0000 | ≤ numerical tolerance | Pass |

All declared coefficient relations pass. The independently evaluated ordering $B_{PA}\leq B_{AI}$ passes because `−0.2273 ≤ −0.1821`. Full left/right values, relationship type, numerical slack, and decisions are saved in `constraint_checks.csv`. No fitted sign constraint is active, and the constrained estimates equal unconstrained OLS in this run; the reported classical standard errors and intervals therefore refer to that common interior solution.

These diagnostics verify coefficient recovery, arithmetic, and controller behavior within the declared controlled simulation. Operational deployment still requires calibration and external validation on representative measurements.

## Operating ranges

The following values are project engineering gates for the Phase 6 simulation, not universal security standards:

| Variable | Preferred | Managed | Critical but modeled | True out-of-domain | Required response |
|---|---|---|---|---|---|
| AI | 0–0.167 | >0.167–0.333 | >0.333–1.000 | >1.000 (`epsilon > 0.30`) | Increase controls in the critical band; reject without prediction outside the fitted domain |
| PA | 0–0.20 | >0.20–0.40 | >0.40–0.50 | >0.50 | Escalate leakage investigation; reject without prediction outside the fitted domain |
| DM | Selected to meet Y ≥0.70 | Selected to meet Y ≥0.60 | DM=1 when target is infeasible | <0 or >1 | Apply mapped defense settings or reject invalid input |
| GS | 0.75–1.00 (4–5 clients) | 0.50 (3 clients) | 0–0.25 (1–2 clients, not deployable) | <0 or >1 | Require at least three clients |
| Y | ≥0.70 | 0.60–<0.70 | <0.60 | <0 or >1 | Reject or defer below 0.60 |

The stress test demonstrates both cases. At modeled `AI=0.6667` and `PA=0.50`, maximum `DM=1.0` and `GS=1.0` predict only `Y=0.5865`, so deployment is rejected or deferred. At `epsilon=0.35` and `PA=0.60`, the state is outside the fitted domain; the code returns no prediction and rejects it.

## Federated optimization values

For each round, Phase 6 searches `DM = 0.00, 0.05, …, 1.00` and three to five deployable active clients (`GS = 0.50, 0.75, 1.00`). It minimizes:

$$
J_t=0.60DM_t^2+0.40GS_t^2+0.15(DM_t-DM_{t-1})^2+0.10(GS_t-GS_{t-1})^2
$$

subject to:

$$
\widehat{Y}_{t+1}\geq0.60,\quad 0\leq DM_t\leq1,\quad GS_t\in\{0.50,0.75,1.00\}.
$$

The first two terms limit defense and aggregation resource use; the last two discourage abrupt configuration changes. The controller first seeks preferred `Y ≥ 0.70`, then minimum `Y ≥ 0.60`. If neither is feasible, it reports `maximum_controls_target_unmet` and rejects or defers the round.

The continuous defense setting is operationally mapped as `adversarial fraction = 0.50 × DM`, `training epsilon = 0.05 × DM`, and `gradient-noise standard deviation = 0.0003 × DM`. At `DM=1`, these match the combined Phase 4 settings. This mapping is a controller policy and does not constitute a formal differential-privacy guarantee.

The distinction between optimized and fixed federated values is:

| Value | Current setting | Phase 6 treatment |
|---|---:|---|
| Defense intensity | 0.00–1.00 in 0.05 steps | Optimized each round |
| Participating clients | 3, 4, or 5 | Optimized each round |
| Total available clients | 5 | Fixed |
| Federated rounds | 12 | Fixed from the saved Phase 2 run |
| Local epochs | 2 per round | Fixed from the saved Phase 2 run |
| Batch size | 64 | Fixed from the saved Phase 2 run |
| Adam learning rate | 0.001 | Fixed in `phase2_federated_learning.py` |

Therefore, Phase 6 is a defense-and-participation optimizer; it does not claim to optimize the learning rate, local epochs, batch size, or stopping round.

A sensitivity check repeats the managed state `AI=0.3333, PA=0.30` under three cost policies. The balanced policy selects `DM=0.80` with four clients; making defense more expensive selects `DM=0.65` with five clients; making participation more expensive selects `DM=1.00` with three clients. All three reach `Y ≥ 0.60`. This shows that cost weights affect resource allocation and are declared policy choices, not coefficients learned from the data.

## Evidence artifacts

Running Phase 6 writes only to `simulation_evidence/constrained_dynamic_model/`:

- `model_fit.json` — equation, coefficients, diagnostics, estimator status, and seed
- `coefficient_estimates.csv` — estimates, standard errors, p-values, and confidence intervals
- `constraint_checks.csv` — every inequality, its operands, slack, and decision
- `simulation_observations.csv` — all 700 estimation observations
- `dynamic_optimization_schedule.csv` — twelve round-specific controller decisions
- `cost_weight_sensitivity.csv` — control choices under three resource-cost policies
- `acceptance_ranges.csv` — ideal, managed, and outside-range policies
- `stress_test_scenarios.csv` — feasible and rejected deployment examples

Run the automated verification with:

```powershell
python -m unittest tests.test_constrained_dynamic_model -v
```

Professional documentation is generated from the same artifacts:

- [Word report](docs/constrained_dynamic_model/DO_FMBS_CONSTRAINED_DYNAMIC_MODEL_REPORT.docx)
- [PDF report](docs/constrained_dynamic_model/DO_FMBS_CONSTRAINED_DYNAMIC_MODEL_REPORT.pdf)
- [Readable Markdown report](docs/constrained_dynamic_model/DO_FMBS_CONSTRAINED_DYNAMIC_MODEL_REPORT.md)

Regenerate all three with `python scripts/generate_constrained_dynamic_report.py`.

## Requirements

- Python 3.9–3.11 (3.10 recommended)
- 8 GB RAM minimum; 16 GB recommended
- CPU execution is supported; a compatible GPU is optional for training phases
- Install dependencies with `pip install -r requirements_app.txt`
