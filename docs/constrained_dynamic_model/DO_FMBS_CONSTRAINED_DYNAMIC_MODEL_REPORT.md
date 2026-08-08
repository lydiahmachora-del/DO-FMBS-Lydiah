# DO-FMBS Phase 6: Constrained Dynamic Optimization Model

## Professional simulation, coefficient-estimation and deployment report

| Report field | Value |
| --- | --- |
| Researcher | Lydiah Moraa Machora |
| Registration number | I152/P/9515/23 |
| Institution | Jaramogi Oginga Odinga University of Science and Technology |
| Research year | 2026 |
| Evidence decision | **PASS** |
| Evidence snapshot generated | 2026-08-08T14:55:24+00:00 |
| Estimator | constrained least squares (SLSQP) |
| Controlled observations | 700 |
| Random seed | 42 |

## Executive conclusion

The Phase 6 constrained model passed the project’s declared simulation-fit gates. The constrained optimization converged, all four fitted sign restrictions and all five independently evaluated ordering hypotheses were satisfied, R² was 0.9681, five-fold cross-validated R² was 0.9673, and RMSE was 0.0245 on the 0-to-1 outcome scale. The fitted signs align with the model: AI and PA reduce predicted efficiency, while DM and GS increase it.

The dynamic controller accepted 12 of 12 simulated federation rounds: 9 met the preferred target of 0.7 and 3 met the managed minimum target of 0.6. Stress tests show that maximum controls cannot maintain the target under the highest combined AI and PA conditions; those cases are explicitly rejected or deferred.

## 1. Constrained model and variable alignment

The estimated transition equation is:

$$
\widehat{Y}_{t+1}=\widehat{\beta}_0+\widehat{\beta}_{AI}AI_t+\widehat{\beta}_{PA}PA_t+\widehat{\beta}_{DM}DM_t+\widehat{\beta}_{GS}GS_t
$$

All four predictors are normalized, and their fitted domains are declared explicitly. This alignment makes sign, magnitude and operating-range comparisons interpretable. AI and PA are observed threat states at round *t*. DM and GS are controller-selected actions at round *t*. The outcome is the simulated next-round FMBS efficiency score at round *t + 1*.

| Variable | Aligned name | Operational definition | Role | Model domain | Desired direction |
| --- | --- | --- | --- | --- | --- |
| AI | Adversarial Input | FGSM perturbation strength normalized as ε / 0.30 | Observed state | 0.000–1.000 (epsilon 0–0.30) | Lower |
| PA | Privacy Attack | Twice the membership-inference accuracy advantage above chance | Observed state | 0.00–0.50 | Lower |
| DM | Defense Mechanism | Deployable defense-policy intensity selected by the controller | Control | 0.00–1.00 | Higher when threat rises |
| GS | Global Server | Normalized client participation: (K − 1) / 4 | Control | 0.00–1.00 (1–5 clients) | Higher participation |
| Y | FMBS efficiency | Simulated next-round efficiency score | Outcome | 0.00–1.00 | >= 0.7 preferred |

PA is an attack-advantage measure, not a general privacy guarantee. In this fitted design PA extends from 0 to 0.50. GS is a participation ratio, not an aggregation coefficient and not a count of training epochs. DM is a normalized control intensity so that a single β_DM coefficient is meaningful.

## 2. Estimated beta values and statistical evidence

| Parameter | Generating value | Estimate | Recovery error | Standardized | SE | 95% CI | p-value |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| B0 | 0.4 | 0.400738 | 0.000738 | 0 | 0.00246 | [0.395909, 0.405567] | < 1×10⁻³⁰⁰ (numerical underflow) |
| B_AI | -0.18 | -0.182145 | -0.002145 | -0.4467 | 0.002764 | [-0.187572, -0.176718] | 3.957e-301 |
| B_PA | -0.22 | -0.227329 | -0.007329 | -0.3068 | 0.005023 | [-0.237192, -0.217466] | 2.241e-209 |
| B_DM | 0.26 | 0.261393 | 0.001393 | 0.7112 | 0.002491 | [0.256501, 0.266284] | < 1×10⁻³⁰⁰ (numerical underflow) |
| B_GS | 0.16 | 0.159417 | -0.000583 | 0.4107 | 0.002632 | [0.15425, 0.164584] | 1.648e-279 |

- **B0 = 0.400738:** Expected simulated next-round FMBS efficiency when AI, PA, DM and GS are zero.
- **B_AI = -0.182145:** Change in predicted efficiency for a one-unit rise in normalized adversarial-input pressure, holding other variables fixed.
- **B_PA = -0.227329:** Change in predicted efficiency for a one-unit rise in membership-inference advantage, holding other variables fixed.
- **B_DM = 0.261393:** Change in predicted efficiency for a one-unit rise in defense intensity, holding other variables fixed.
- **B_GS = 0.159417:** Change in predicted efficiency for a one-unit rise in the participating-client ratio, holding other variables fixed.

On the standardized scale, **B_DM** has the largest absolute association (0.7112). This ranking describes the simulated design; it is not a claim of universal causal importance.

The largest absolute coefficient-recovery error was **0.007329**. The constrained and unconstrained solutions differed by at most **0**; active fitted constraints: **none**. Inference basis: Classical OLS covariance is reported because the constrained solution equals the unconstrained OLS solution and no fitted sign constraint is active.

### Coefficient-constraint verification

| Constraint | Type | Left | Right | Slack | Decision |
| --- | --- | ---: | ---: | ---: | :---: |
| B_AI < B_DM | Evaluated hypothesis | -0.182145 | 0.261393 | 0.443537 | Passed |
| B_AI < B_GS | Evaluated hypothesis | -0.182145 | 0.159417 | 0.341562 | Passed |
| B_PA <= B_AI | Evaluated hypothesis | -0.227329 | -0.182145 | 0.045184 | Passed |
| B_PA < B_DM | Evaluated hypothesis | -0.227329 | 0.261393 | 0.488722 | Passed |
| B_PA < B_GS | Evaluated hypothesis | -0.227329 | 0.159417 | 0.386746 | Passed |
| B_AI <= 0 | Fitted sign restriction | -0.182145 | 0 | 0.182145 | Passed |
| B_PA <= 0 | Fitted sign restriction | -0.227329 | 0 | 0.227329 | Passed |
| B_DM >= 0 | Fitted sign restriction | 0.261393 | 0 | 0.261393 | Passed |
| B_GS >= 0 | Fitted sign restriction | 0.159417 | 0 | 0.159417 | Passed |

Maximum numerical constraint violation: **0**. Estimator termination: **Optimization terminated successfully**.

## 3. Model-fit diagnostics and decision ranges

The following thresholds are project reporting gates for a normalized simulation model. They are declared decision rules, not universal standards for every biometric deployment.

| Diagnostic | Attained value | Ideal / pass | Managed | Outside | Decision |
| --- | ---: | :---: | :---: | :---: | --- |
| R² | 0.9681 | >=0.90 | 0.75–<0.90 | <0.75 | Ideal / pass |
| Adjusted R² | 0.9679 | >=0.90 | 0.75–<0.90 | <0.75 | Ideal / pass |
| Five-fold CV R² | 0.9673 ± 0.0015 | >=0.90 | 0.75–<0.90 | <0.75 | Ideal / pass |
| RMSE | 0.0245 | <=0.05 | >0.05–0.10 | >0.10 | Ideal / pass |
| MAE | 0.0196 | <=0.05 | >0.05–0.10 | >0.10 | Ideal / pass |
| Maximum VIF | 1 | <5 | 5–10 | >10 | Ideal / pass |
| Condition number | 7.2485 | <30 | 30–100 | >100 | Ideal / pass |
| Durbin–Watson | 1.8678 | Recorded only | Not rated | Not rated | Not interpreted: factorial row order is non-temporal |
| Shapiro–Wilk p | 0.6423 | >=0.05 | 0.01–<0.05 | <0.01 | Ideal / pass |

Residual mean was 4.230e-16 and residual standard deviation was 0.02455. Cross-validated RMSE was 0.02471. The diagnostics are internally consistent with a well-conditioned linear approximation for the simulated domain.

## 4. Declared operating bands and outside-range decisions

| Variable | Definition | Fitted domain | Preferred | Managed | Critical but modeled | Out of domain | Required action |
| --- | --- | --- | --- | --- | --- | --- | --- |
| AI | epsilon / 0.30 | 0.000–1.000 (epsilon 0–0.30) | 0.000–0.167 | >0.167–0.333 | >0.333–1.000 | >1.000 | Increase controls in the critical band; reject without prediction outside the fitted domain. |
| PA | 2 × max(0, membership-attack accuracy − 0.50) | 0.00–0.50 | 0.00–0.20 | >0.20–0.40 | >0.40–0.50 | >0.50 | Escalate leakage investigation; reject without prediction outside the fitted domain. |
| DM | continuous defense policy intensity | 0.00–1.00 | controller-selected to meet Y >= 0.70 | controller-selected to meet Y >= 0.60 | maximum DM=1 when target is infeasible | <0 or >1 | Apply the mapped defense settings; reject invalid values. |
| GS | (participating clients − 1) / 4 | 0.00–1.00 (1–5 clients) | 0.75–1.00 (4–5 clients) | 0.50 (3 clients) | 0.00–0.25 (1–2 clients; not deployable) | <0 or >1 | Require at least three clients; otherwise defer aggregation. |
| Y | simulated next-round FMBS efficiency score | 0.00–1.00 | >=0.70 | 0.60–<0.70 | <0.60 | <0 or >1 | Reject or defer a round below 0.60. |

These are operating-policy bands. A value in a managed band may continue only when the controller finds controls that keep predicted Y at or above the minimum target. A critical value remains within the fitted simulation domain but requires maximum escalation and can still lead to rejection or deferral. A value outside its variable-specific fitted domain is rejected without producing a model prediction; it is never silently clipped during deployment.

### Stress-test decisions

| Scenario | ε | AI | PA | DM | GS | Clients | Predicted Y | Status | Deployment decision |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |
| Preferred operating | 5.000e-02 | 0.167 | 0.1 | 0.9 | 0.75 | 4 | 0.7025 | Preferred target met | Accept |
| Managed operating | 0.1 | 0.333 | 0.3 | 0.8 | 0.75 | 4 | 0.6005 | Minimum target met | Accept |
| Modeled critical | 0.2 | 0.667 | 0.5 | 1 | 1 | 5 | 0.5865 | Maximum controls; target unmet | Reject or defer |
| Out of domain | 0.35 | 1.167 | 0.6 | Not evaluated | Not evaluated | Not selected | Not evaluated | Out of domain | Reject or defer |

At the modeled-critical condition (AI=0.67, PA=0.5), the controller selected maximum DM and GS but predicted Y=0.5865, below the 0.6 target. The recorded decision is **Reject or defer**. This is the intended fail-safe behavior when additional control cannot restore the required operating level.

The out-of-domain scenario supplied ε=0.35 (AI=1.1667) and PA=0.6. Because both exceed their fitted domains, no DM, GS or predicted Y was produced; the recorded decision is **Reject or defer**.

## 5. Numerical substitution using estimated coefficients

Round 6 is used because it is the highest combined AI and PA condition in the dynamic schedule that still meets the target. Its values are:

AI=0.666667, PA=0.4, DM=1, GS=1.

Substitute the estimated beta values:

$$
\widehat{Y}_{t+1}=0.400738+(-0.182145)(0.666667)+(-0.227329)(0.4)+(0.261393)(1)+(0.159417)(1)
$$

| Term | Contribution |
| --- | ---: |
| B0 | 0.400738 |
| B_AI × AI | -0.12143 |
| B_PA × PA | -0.090932 |
| B_DM × DM | 0.261393 |
| B_GS × GS | 0.159417 |
| Total predicted Y | 0.609186 |

The recomputed result is **0.609186**; the stored optimizer value is **0.609186**. Their absolute difference is **0**, confirming that the saved schedule uses the estimated equation.

## 6. What is dynamic and how optimization occurs

The beta coefficients are estimated once for this evidence snapshot; the controls are dynamic. Before every simulated federation round, the controller observes current AI_t and PA_t. It then re-evaluates all deployable DM_t and GS_t combinations, considers the previous round’s controls, and selects a new policy for predicting Y_(t+1). Therefore, dynamic means state-dependent, round-by-round adaptation—not coefficients that are arbitrarily changed during inference.

### 6.1 Coefficient-estimation optimization

The estimator solves:

$$
\min_{\beta}\;\frac{1}{2}\lVert X\beta-y\rVert_2^2
$$

subject to β_AI ≤ 0, β_PA ≤ 0, β_DM ≥ 0 and β_GS ≥ 0. The five coefficient-order relations shown in the constraint table are independently evaluated hypotheses; they are not enforced during estimation. SciPy SLSQP reported: **Optimization terminated successfully**.

### 6.2 Dynamic controller objective and exact search

For every candidate control pair, the controller calculates:

$$
C_t=0.6DM_t^2+0.4GS_t^2+0.15(DM_t-DM_{t-1})^2+0.1(GS_t-GS_{t-1})^2
$$

It enumerates DM from 0.00 to 1.00 in 0.05 steps (21 values) and participating clients K ∈ {3, 4, 5}, where GS=(K−1)/4. This gives **63 deployable candidates per round**. The first round starts from DM_(t−1)=0.30 and GS_(t−1)=0.75.

The decision rule first searches for candidates satisfying preferred Y >= 0.7. If none exist, it searches for the managed minimum Y >= 0.6. Within the highest feasible tier it minimizes C_t; a cost tie is resolved in favour of higher predicted Y. If no candidate meets even the minimum target, it selects the candidate with the highest predicted Y and marks the round `maximum_controls_target_unmet`. This is a finite, reproducible constrained grid search rather than an unspecified claim of optimization.

### 6.3 Cost-weight sensitivity

The cost weights express a deployment policy; they are not estimated beta coefficients. Phase 6 therefore repeats the same managed threat state under three cost profiles to show how the selected control mix changes.

| Cost profile | w_DM | w_GS | DM | GS | Clients | Predicted Y | Cost | Result |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| Balanced | 0.6 | 0.4 | 0.8 | 0.75 | 4 | 0.6005 | 0.6465 | Minimum target met |
| Defense sparing | 1 | 0.2 | 0.65 | 1 | 5 | 0.6011 | 0.6471 | Minimum target met |
| Participation sparing | 0.3 | 0.8 | 1 | 0.5 | 3 | 0.6129 | 0.5797 | Minimum target met |

At AI=0.333 and PA=0.30, the balanced profile selected DM=0.80 with four clients; the defense-sparing profile selected DM=0.65 with five clients; and the participation-sparing profile selected DM=1.00 with three clients. All three met the managed minimum Y target of 0.6. The switching weights remained 0.15 for DM and 0.10 for GS in all profiles. This sensitivity result shows that cost weights change the chosen feasible policy, so they must be selected and disclosed as governance inputs.

## 7. Federation-learning control values

Within Phase 6, the federated control is client participation. Five clients are available; GS normalizes participation as (K−1)/4. The controller requires at least three clients, so the deployable GS candidates are 0.50, 0.75 and 1.00. It does not reinterpret GS as a model weight. Once clients are selected, sample-weighted federated aggregation remains a separate training operation.

### Fixed versus optimized federation settings

| Setting | Value/search space | Phase 6 role | Interpretation |
| --- | --- | --- | --- |
| Defense intensity (DM) | 0.00–1.00 in 0.05 steps | Optimized by Phase 6 | Controls adversarial fraction, training ε and gradient-noise standard deviation |
| Active clients (K) / GS | K ∈ {3, 4, 5}; GS=(K−1)/4 | Optimized by Phase 6 | Minimum three active clients |
| Federated rounds | 12 | Fixed integration setting | Not searched by Phase 6 |
| Local epochs per round | 2 | Fixed integration setting | Not searched by Phase 6 |
| Batch size | 64 | Fixed integration setting | Not searched by Phase 6 |
| Learning rate | 1.000e-03 | Fixed integration setting | Not searched by Phase 6 |
| Total available clients | 5 | Fixed integration setting | Not searched by Phase 6 |

The learning rate, local-epoch count, batch size and federated-round count are fixed at the values above; Phase 6 does **not** claim to optimize them. Its adaptive decisions are limited to DM and the active-client choice represented by GS.

| Round | ε | AI | PA | DM | GS | Clients | Predicted Y | Cost | Status |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| 1 | 0 | 0 | 0.02 | 0.75 | 0.75 | 4 | 0.7118 | 0.5929 | Preferred target met |
| 2 | 1.000e-02 | 0.033 | 0.02 | 0.75 | 0.75 | 4 | 0.7057 | 0.5625 | Preferred target met |
| 3 | 5.000e-02 | 0.167 | 0.05 | 0.85 | 0.75 | 4 | 0.7008 | 0.66 | Preferred target met |
| 4 | 0.1 | 0.333 | 0.1 | 0.9 | 1 | 5 | 0.712 | 0.8926 | Preferred target met |
| 5 | 0.15 | 0.5 | 0.2 | 0.85 | 0.75 | 4 | 0.6059 | 0.6651 | Minimum target met |
| 6 | 0.2 | 0.667 | 0.4 | 1 | 1 | 5 | 0.6092 | 1.0096 | Minimum target met |
| 7 | 0.1 | 0.333 | 0.3 | 0.8 | 0.75 | 4 | 0.6005 | 0.6213 | Minimum target met |
| 8 | 5.000e-02 | 0.167 | 0.15 | 0.95 | 0.75 | 4 | 0.7042 | 0.7699 | Preferred target met |
| 9 | 1.000e-02 | 0.033 | 0.08 | 0.8 | 0.75 | 4 | 0.7052 | 0.6124 | Preferred target met |
| 10 | 0 | 0 | 0.02 | 0.75 | 0.75 | 4 | 0.7118 | 0.5629 | Preferred target met |
| 11 | 5.000e-02 | 0.167 | 0.1 | 0.9 | 0.75 | 4 | 0.7025 | 0.7144 | Preferred target met |
| 12 | 0.1 | 0.333 | 0.2 | 0.95 | 1 | 5 | 0.7023 | 0.9481 | Preferred target met |

### Defense settings produced by optimized DM

| DM | Adversarial fraction | Training attack ε | Gradient-noise SD |
| ---: | ---: | ---: | ---: |
| 0.75 | 0.375 | 0.0375 | 0.000225 |
| 0.8 | 0.4 | 0.04 | 0.00024 |
| 0.85 | 0.425 | 0.0425 | 0.000255 |
| 0.9 | 0.45 | 0.045 | 0.00027 |
| 0.95 | 0.475 | 0.0475 | 0.000285 |
| 1 | 0.5 | 0.05 | 0.0003 |

The mapping is linear: adversarial fraction = 0.50×DM, training attack ε = 0.05×DM, and Gaussian gradient-noise standard deviation = 0.0003×DM. These are simulation-policy controls and do not by themselves establish formal differential privacy.

Across 12 rounds, DM ranged from 0.75 to 1.00; GS ranged from 0.75 to 1.00; participating clients ranged from 4 to 5. Client selections were K=4: 9 round(s), K=5: 3 round(s). The preferred target was met in 9/12 rounds; the remaining 3 accepted round(s) met the managed minimum.

Optimizing rounds, local epochs, batch size or learning rate would require a separate training-cost experiment and is outside this controller’s declared search space.

## 8. Simulation design and evidence provenance

Design declared by the fit artifact: **7 AI x 5 PA x 4 DM x 5 GS = 700 balanced observations**.

The simulation-generating coefficients are known design parameters used to test whether the estimator can recover the intended constrained relationship. They are not fitted results. The estimated coefficients and their recovery errors are reported separately in Section 2.

| Design field | Levels | Values | Minimum | Maximum |
| --- | ---: | --- | ---: | ---: |
| epsilon | 7 | 0, 0.01, 0.05, 0.1, 0.15, 0.2, 0.3 | 0 | 0.3 |
| AI | 7 | 0, 0.033, 0.167, 0.333, 0.5, 0.667, 1 | 0 | 1 |
| PA | 5 | 0, 0.02, 0.1, 0.25, 0.5 | 0 | 0.5 |
| DM | 4 | 0, 0.33, 0.67, 1 | 0 | 1 |
| GS | 5 | 0, 0.25, 0.5, 0.75, 1 | 0 | 1 |

The observation file contains 700 rows. Simulation errors have mean -0.000960, sample standard deviation 0.024600, minimum -0.074113, and maximum 0.072847. The fixed seed is 42.

### Evidence manifest

| Artifact | Content | Bytes | SHA-256 |
| --- | --- | ---: | --- |
| model_fit.json | 1 JSON object | 4313 | 384a1ae365de311ba9b0f2fd3a37eb361c8f49b6300c93c4e037b1d22639f2c8 |
| acceptance_ranges.csv | 5 data rows | 1083 | d5bf6c3e5a715ae37fee3d742825f6409ca0301d370611cdd7d1db34b7791f8d |
| coefficient_estimates.csv | 5 data rows | 1183 | ab83724d8d5ffba92d06735b63a1045a84dd4423c0e9ee23983a18d9398a031e |
| constraint_checks.csv | 9 data rows | 880 | 41f432c4c8a5e2f797eb80df8906bc3efb1d7963217243c6e84270791a6c9390 |
| cost_weight_sensitivity.csv | 3 data rows | 696 | 3deb16631b239cd8ad45d4cd6184427b774fb53aba8b08c68114c2bed558be11 |
| dynamic_optimization_schedule.csv | 12 data rows | 1882 | fd71db55b7b7de4767bfc25aae29e06f6adea6b6f065601a2f7802ad55105211 |
| simulation_observations.csv | 700 data rows | 56745 | c761aa59c6c0f33c139cc8397387b3acf3640351f417e8360226d1bd9cc2bd75 |
| stress_test_scenarios.csv | 4 data rows | 701 | d41dbcc56410d4b0027666d98a9c433ad66092301a95f4fdcf6b104cd825c78a |

Repository commit at report generation: `31c61eb115ca61320c3af7eb293264e6787febb8`

Python runtime: `3.11.9` on `Windows-10-10.0.19045-SP0`

Report generator: `scripts/generate_constrained_dynamic_report.py`

## 9. Equation-to-code and artifact traceability

The table below identifies the functions that produce each mathematical component. It allows an examiner or reviewer to move directly from the equation and variable definitions to the executable implementation and saved evidence.

| Model element | Implemented operation | Function(s) | Location / evidence |
| --- | --- | --- | --- |
| AI, PA and GS alignment | Normalize ε, calculate membership advantage, and map participating clients to GS=(K−1)/4 | `normalize_ai`, `membership_advantage`, `gs_from_clients` | `scripts/constrained_dynamic_model.py`; `acceptance_ranges.csv` |
| DM operational mapping | Map defense intensity to adversarial fraction, training attack ε and gradient-noise SD | `defense_configuration` | `scripts/constrained_dynamic_model.py`; schedule and sensitivity CSVs |
| Simulated transition | Generate Y_(t+1) from AI_t, PA_t, DM_t and GS_t plus controlled error | `simulation_response`, `generate_observations` | `scripts/constrained_dynamic_model.py`; `simulation_observations.csv` |
| Beta estimation | Minimize one-half residual sum of squares subject to four fitted sign restrictions | `_constrained_coefficients`, `fit_equation` | `scripts/constrained_dynamic_model.py`; `coefficient_estimates.csv`; `model_fit.json` |
| Equation substitution | Calculate predicted Y using the five estimated beta values | `predict` | `scripts/constrained_dynamic_model.py`; dynamic and stress CSVs |
| Dynamic optimization | Search DM and active-client controls at every round under preferred/minimum targets | `optimize_controls`, `dynamic_schedule`, `cost_weight_sensitivity` | `scripts/constrained_dynamic_model.py`; schedule and sensitivity CSVs |
| Bands and fail-safe decisions | Declare fitted/preferred/managed/critical bands and reject out-of-domain or infeasible cases | `acceptance_ranges`, `stress_scenarios` | `scripts/constrained_dynamic_model.py`; acceptance and stress CSVs |
| Artifact production | Run estimation, optimization and CSV/JSON serialization | `main` | `phase6_constrained_dynamic_optimization.py`; `simulation_evidence/constrained_dynamic_model/` |

## 10. Interpretation and deployment decision protocol

For each proposed federation round:

1. Measure ε and calculate AI = ε / 0.30; estimate PA using the defined membership-advantage procedure.
2. Reject malformed values outside their fitted domains: AI outside 0–1 or PA outside 0–0.50.
3. Search the 63 allowed DM/client-participation combinations.
4. Prefer Y >= 0.7; otherwise accept managed operation only when Y >= 0.6 and all controls are available.
5. If the target is infeasible, isolate the affected input or defer aggregation; do not report the maximum-control result as acceptable.
6. Treat Y >= 0.7 as preferred and 0.6 <= Y < 0.7 as managed operation requiring monitoring.

## 11. Plain-language oral explanation

This phase creates many controlled situations in which attack pressure, privacy pressure, defense strength and client participation are varied. It uses those results to estimate how each variable changes the next-round FMBS efficiency score. The negative beta values show that AI and PA reduce the score, while the positive beta values show that DM and GS improve it. The model is dynamic because it measures the current threats and chooses new controls before every federation round. It first tries to reach the preferred score of 0.70 and otherwise allows managed operation only at or above 0.60. It searches defense intensity and three-to-five active clients, while rounds, local epochs, batch size and learning rate remain fixed. If maximum controls cannot reach 0.60, or if an input is outside the fitted domain, the round is rejected or deferred.

## 12. Limitations and responsible claims

- The coefficient estimates validate recovery and constrained optimization on controlled simulated observations; external operational data are still required for field validation.
- Confidence intervals and p-values are conditional on the specified linear model and its error assumptions. They do not establish causality.
- The operating bands are declared project controls, not universal biometric-industry thresholds.
- The controller must not extrapolate beyond the normalized input domains. Values outside those domains require measurement correction or rejection.
- GS optimizes participating-client count only, with at least three active clients. It does not optimize local epochs, learning rate, batch size, round count, communication latency or client-specific aggregation weights.
- PA quantifies the selected membership-inference advantage; a low PA value does not establish protection against every privacy attack.
- Y is a simulated next-round FMBS efficiency score. A deployment assessment must additionally validate identity-specific accuracy, conventional biometric FAR/FRR, calibration, subgroup performance, privacy accounting and adaptive attacks.
- Scope statement from the evidence artifact: This is a reproducible synthetic constrained-coefficient recovery and adaptive-policy demonstration. Coefficients are simulation estimates, not neural-network weights or estimates from 700 independent federated training runs.
- A high R² demonstrates fit to this simulated design. It does not by itself establish safety, fairness, privacy or generalization to a new population.

## 13. Professional conclusion

The constrained equation was estimated from 700 controlled observations, not populated by assigning outcome metrics to beta symbols. All four fitted sign restrictions and all five independently evaluated ordering hypotheses passed, the constrained estimator converged, and the attained R² of 0.9681 was supported by cross-validated R² of 0.9673. The dynamic controller then used those estimates to choose defense intensity and participating-client ratio at every round. It met the minimum Y target in 12/12 dynamic rounds (9 preferred, 3 managed-minimum) and correctly rejected or deferred stress conditions for which maximum controls were insufficient.

Accordingly, the mathematical computation, constraint enforcement and simulated control logic are internally verified for the declared operating domain. Operational deployment remains conditional on prospective validation using representative biometric and federation data.
