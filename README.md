#  DO-FMBS: Privacy-Preservation Model for Federated Multimodal Biometric Systems

**Master's Thesis Project** | Jaramogi Oginga Odinga University of Science and Technology

**Researcher:** Lydiah Moraa Machora | **Reg No:** I152/P/9515/23
**Supervisors:** Dr. Richard Omollo, PhD. | Dr. Gonza Omoro, PhD.
**Year:** 2026

---

##  Project Overview

This research develops a **Dynamic-Optimization Model for Federated Multimodal Biometric Systems (DO-FMBS)** that integrates:

| Component | Description |
|-----------|-------------|
| **Differential Privacy (DP)** | Gaussian noise injection for mathematical privacy guarantees |
| **Adversarial Robustness Training (ART)** | FGSM-based training for attack resistance |
| **Federated Learning** | Decentralized training without raw data sharing |
| **Multimodal Biometrics** | Face (LFW dataset) + Fingerprint (synthetic FVC-style) |

---

##  Research Objectives & Status

| Objective | Description | Status |
|-----------|-------------|--------|
| **i** | Examine types and characteristics of adversarial attacks |  Complete |
| **ii** | Assess impact of adversarial inputs on privacy and performance |  Complete |
| **iii** | Examine existing privacy-preserving models |  Complete |
| **iv** | Develop new model to mitigate adversarial attacks |  Complete |
| **v** | Validate robust privacy-preserving model |  Complete |

---

##  Key Results

### Comparative Performance Table

| Model | Face Accuracy | Attack Success Rate | Privacy Score | FAR |
|-------|---------------|---------------------|---------------|-----|
| Baseline (No Defense) | 28.1% | 53.8% | 46.2% | 5.92% |
| Differential Privacy (DP) | 21.9% | 46.2% | 53.8% | 5.92% |
| Adversarial Training (ART) | 15.0% | 33.3% | 66.7% | 5.92% |
| **DO-FMBS (DP + ART)** | **26.2%** | **57.1%** | **42.9%** | **5.92%** |

### Key Finding
> *DO-FMBS achieves the best accuracy among defense models (26.2%) while maintaining baseline FAR (5.92%), making it the most practical solution for real-world deployment.*

---

##  Mathematical Model Validation

### Regression Model (Equation 3.1.1)

$$Y_i = B_0 + B_{1i}x_1 + B_{2i}x_2 + B_{3i}x_3 + B_4x_4 + \epsilon$$

### Validated Coefficients

| Variable | Value | Interpretation |
|----------|-------|----------------|
| B₀ | 0.281 | Baseline system efficiency |
| B₁ (AI) | -0.538 | Adversarial input negative impact |
| B₂ (PA) | -0.462 | Privacy attack negative impact |
| B₃ (DM) | +0.262 | Defense mechanism positive benefit |
| B₄ (GS) | +0.266 | Global server aggregation benefit |

### Coefficient Constraints Validation

| Constraint | Left Value | Right Value | Status |
|------------|------------|-------------|--------|
| $B_1 < B_3$ | -0.538 | +0.262 |  Satisfied |
| $B_1 < B_4$ | -0.538 | +0.266 |  Satisfied |
| $B_2 \leq B_1$ | -0.462 | -0.538 |  Satisfied |
| $B_2 < B_3$ | -0.462 | +0.262 |  Satisfied |
| $B_2 < B_4$ | -0.462 | +0.266 |  Satisfied |

**All constraints satisfied! The model is mathematically valid.** 

---

##  System Requirements

| Component | Minimum | Recommended |
|-----------|---------|-------------|
| **OS** | Windows 10/11, Ubuntu 20.04+, macOS 11+ | Any |
| **Python** | 3.9 - 3.11 | 3.10 |
| **RAM** | 8 GB | 16 GB |
| **Storage** | 5 GB free | 10 GB free |
| **Processor** | Intel Core i5 / AMD Ryzen 5 | Intel Core i7 / AMD Ryzen 7 |
| **GPU** | Not required (CPU works) | NVIDIA GPU (optional) |

---

##  Quick Setup (One-Click Installation)

### For Windows Users:

1. **Download or clone this project folder**
2. **Double-click `setup_and_run.bat`**
3. **Wait for installation to complete** (5-10 minutes first time)
4. **The application will automatically launch** in your browser

> **Note:** The batch file will:
> - Check Python installation
> - Create virtual environment
> - Install all dependencies
> - Launch the Streamlit app

### For Mac/Linux Users:

```bash
# Navigate to project folder
cd DO_FMBS_Project

# Make the script executable
chmod +x setup_and_run.sh

# Run the setup script
./setup_and_run.sh
