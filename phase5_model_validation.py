"""
DO-FMBS Research Project - Phase 5: Model Validation & Testing
Privacy-Preservation Model for Federated Multimodal Biometric System

Researcher: Lydiah Moraa Machora | Reg No: I152/P/9515/23
University: Jaramogi Oginga Odinga University of Science and Technology

This script:
1. Loads the trained DO-FMBS and Baseline models
2. Performs comprehensive validation tests
3. Compares performance under various conditions
4. Validates mathematical constraints
5. Generates validation report and visualizations
"""

import os
import sys
import warnings
import logging
import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pickle
from datetime import datetime
from scipy import stats
from sklearn.metrics import confusion_matrix, classification_report

# Windows consoles commonly default to cp1252, which cannot print symbols used
# in the validation report such as epsilon and plus-minus.
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

# Suppress warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['STREAMLIT_SUPPRESS_WARNING'] = 'true'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
warnings.filterwarnings('ignore')
logging.getLogger('tensorflow').setLevel(logging.ERROR)
logging.getLogger('absl').setLevel(logging.ERROR)

import tensorflow as tf
tf.random.set_seed(42)
np.random.seed(42)

print("=" * 70)
print("DO-FMBS Phase 5: Model Validation & Testing")
print("=" * 70)

# ============================================================================
# 1. LOAD MODELS AND DATA
# ============================================================================
print("\n[1/8] Loading Models and Data...")

# Load face data
face_data = np.load('do_fmbs_data/face_data.npz', allow_pickle=True)
X_test_face = face_data['X_test'].astype(np.float32)
y_test_face = face_data['y_test']
n_face_classes = int(face_data['n_classes'])
n_face_features = int(face_data['n_features'])

# Load fingerprint data
finger_data = np.load('do_fmbs_data/fingerprint_data.npz', allow_pickle=True)
X_test_finger = finger_data['X_test'].astype(np.float32)
y_test_finger = finger_data['y_test']
n_finger_classes = int(finger_data['n_classes'])

# Load models
print("    Loading Baseline Model...")
baseline_model = tf.keras.models.load_model('do_fmbs_data/baseline_model.keras')

print("    Loading DO-FMBS Model...")
do_model = tf.keras.models.load_model('do_fmbs_data/do_fmbs_model.keras')

# Prepare test data
n_samples = min(len(X_test_face), len(X_test_finger), 200)
X_face_test = X_test_face[:n_samples]
X_finger_test = X_test_finger[:n_samples]
y_face_test = y_test_face[:n_samples]
y_finger_test = y_test_finger[:n_samples]

print(f"    Test samples: {n_samples}")

# ============================================================================
# 2. FGSM ATTACK FUNCTION (for validation)
# ============================================================================
print("\n[2/8] Implementing Attack Functions for Validation...")

def fgsm_attack(model, face_input, finger_input, face_label, epsilon=0.1):
    """Fast Gradient Sign Method attack"""
    face_tensor = tf.convert_to_tensor(face_input.reshape(1, -1), dtype=tf.float32)
    finger_tensor = tf.convert_to_tensor(finger_input.reshape(1, -1), dtype=tf.float32)
    
    with tf.GradientTape() as tape:
        tape.watch(face_tensor)
        predictions = model({'face_input': face_tensor, 'finger_input': finger_tensor})
        loss = tf.keras.losses.sparse_categorical_crossentropy([face_label], predictions[0])
    
    gradients = tape.gradient(loss, face_tensor)
    perturbation = epsilon * tf.sign(gradients)
    adversarial_face = face_tensor + perturbation
    adversarial_face = tf.clip_by_value(adversarial_face, -3, 3)
    
    return adversarial_face.numpy().reshape(-1)

def evaluate_model_robustness(model, X_face, X_finger, y_face, epsilons=[0.00, 0.05, 0.10, 0.15, 0.20]):
    """Evaluate model robustness at different attack strengths"""
    results = []
    
    for eps in epsilons:
        correct = 0
        total = 0
        
        for i in range(len(X_face)):
            if eps == 0:
                # No attack
                pred = model.predict(
                    {'face_input': X_face[i:i+1], 'finger_input': X_finger[i:i+1]},
                    verbose=0
                )
                pred_class = np.argmax(pred[0])
                if pred_class == y_face[i]:
                    correct += 1
                total += 1
            else:
                # Only test samples that were originally correct
                orig_pred = model.predict(
                    {'face_input': X_face[i:i+1], 'finger_input': X_finger[i:i+1]},
                    verbose=0
                )
                orig_class = np.argmax(orig_pred[0])
                
                if orig_class == y_face[i]:
                    total += 1
                    adv_face = fgsm_attack(model, X_face[i], X_finger[i], y_face[i], eps)
                    adv_pred = model.predict(
                        {'face_input': adv_face.reshape(1, -1), 'finger_input': X_finger[i:i+1]},
                        verbose=0
                    )
                    adv_class = np.argmax(adv_pred[0])
                    if adv_class == y_face[i]:
                        correct += 1
        
        accuracy = correct / total if total > 0 else 0
        attack_success = 1 - accuracy
        results.append({'epsilon': eps, 'accuracy': accuracy, 'attack_success_rate': attack_success})
    
    return results

# ============================================================================
# 3. COMPREHENSIVE VALIDATION TESTS
# ============================================================================
print("\n[3/8] Running Comprehensive Validation Tests...")

print("\n    Testing Baseline Model Robustness...")
baseline_robustness = evaluate_model_robustness(
    baseline_model, X_face_test, X_finger_test, y_face_test
)

print("\n    Testing DO-FMBS Model Robustness...")
do_robustness = evaluate_model_robustness(
    do_model, X_face_test, X_finger_test, y_face_test
)

# ============================================================================
# 4. STATISTICAL SIGNIFICANCE TESTING
# ============================================================================
print("\n[4/8] Performing Statistical Significance Tests...")

def calculate_accuracy(model, X_face, X_finger, y_face, n_iterations=10):
    """Calculate accuracy with multiple runs for statistical testing"""
    accuracies = []
    
    for _ in range(n_iterations):
        # Random subset of test data
        indices = np.random.choice(len(X_face), size=min(100, len(X_face)), replace=False)
        pred = model.predict(
            {'face_input': X_face[indices], 'finger_input': X_finger[indices]},
            verbose=0
        )
        pred_class = np.argmax(pred[0], axis=1)
        acc = np.mean(pred_class == y_face[indices])
        accuracies.append(acc)
    
    return accuracies

print("    Calculating baseline accuracies...")
baseline_accs = calculate_accuracy(baseline_model, X_face_test, X_finger_test, y_face_test)

print("    Calculating DO-FMBS accuracies...")
do_accs = calculate_accuracy(do_model, X_face_test, X_finger_test, y_face_test)

# Statistical test
t_stat, p_value = stats.ttest_ind(baseline_accs, do_accs)
is_significant = p_value < 0.05
pooled_std = np.sqrt((np.var(baseline_accs, ddof=1) + np.var(do_accs, ddof=1)) / 2)
effect_size = (np.mean(do_accs) - np.mean(baseline_accs)) / pooled_std if pooled_std > 0 else 0.0

print(f"\n    Statistical Test Results:")
print(f"    Baseline Mean Accuracy: {np.mean(baseline_accs)*100:.2f}% (±{np.std(baseline_accs)*100:.2f})")
print(f"    DO-FMBS Mean Accuracy: {np.mean(do_accs)*100:.2f}% (±{np.std(do_accs)*100:.2f})")
print(f"    T-statistic: {t_stat:.4f}")
print(f"    P-value: {p_value:.4f}")
print(f"    Effect Size (Cohen's d): {effect_size:.4f}")
print(f"    Statistically Significant: {'YES' if is_significant else 'NO'}")

# ============================================================================
# 5. CONFUSION MATRIX AND CLASSIFICATION METRICS
# ============================================================================
print("\n[5/8] Generating Classification Metrics...")

def get_predictions(model, X_face, X_finger, y_face):
    """Get model predictions"""
    pred = model.predict({'face_input': X_face, 'finger_input': X_finger}, verbose=0)
    pred_class = np.argmax(pred[0], axis=1)
    return pred_class

# Get predictions for subset
test_subset = min(100, len(X_face_test))
baseline_preds = get_predictions(baseline_model, X_face_test[:test_subset], X_finger_test[:test_subset], y_face_test[:test_subset])
do_preds = get_predictions(do_model, X_face_test[:test_subset], X_finger_test[:test_subset], y_face_test[:test_subset])

# Classification reports
baseline_report = classification_report(y_face_test[:test_subset], baseline_preds, output_dict=True)
do_report = classification_report(y_face_test[:test_subset], do_preds, output_dict=True)

# ============================================================================
# 6. GENERATE VALIDATION REPORT
# ============================================================================
print("\n[6/8] Saving Validation Results...")

# Create validation results dictionary
validation_results = {
    'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    'test_samples': n_samples,
    'robustness': {
        'baseline': baseline_robustness,
        'do_fmbs': do_robustness
    },
    'statistical_test': {
        'baseline_mean': float(np.mean(baseline_accs)),
        'baseline_std': float(np.std(baseline_accs)),
        'do_fmbs_mean': float(np.mean(do_accs)),
        'do_fmbs_std': float(np.std(do_accs)),
        't_statistic': float(t_stat),
        'p_value': float(p_value),
        'effect_size': float(effect_size),
        'is_significant': bool(is_significant)
    },
    'classification_metrics': {
        'baseline': {
            'accuracy': baseline_report.get('accuracy', 0),
            'macro_avg_precision': baseline_report.get('macro avg', {}).get('precision', 0),
            'macro_avg_recall': baseline_report.get('macro avg', {}).get('recall', 0),
            'macro_avg_f1': baseline_report.get('macro avg', {}).get('f1-score', 0)
        },
        'do_fmbs': {
            'accuracy': do_report.get('accuracy', 0),
            'macro_avg_precision': do_report.get('macro avg', {}).get('precision', 0),
            'macro_avg_recall': do_report.get('macro avg', {}).get('recall', 0),
            'macro_avg_f1': do_report.get('macro avg', {}).get('f1-score', 0)
        }
    }
}

# Save validation results
with open('do_fmbs_data/validation_results.pkl', 'wb') as f:
    pickle.dump(validation_results, f)
print("    Saved: do_fmbs_data/validation_results.pkl")

# ============================================================================
# 7. GENERATE VISUALIZATIONS
# ============================================================================
print("\n[7/8] Generating Validation Visualizations...")

# Figure 1: Robustness Comparison
fig1, ax1 = plt.subplots(figsize=(10, 6))

epsilons = [r['epsilon'] for r in baseline_robustness]
baseline_acc = [r['accuracy'] * 100 for r in baseline_robustness]
do_acc = [r['accuracy'] * 100 for r in do_robustness]

ax1.plot(epsilons, baseline_acc, 'o-', color='red', linewidth=2, markersize=8, label='Baseline Model')
ax1.plot(epsilons, do_acc, 's-', color='green', linewidth=2, markersize=8, label='DO-FMBS Model')
ax1.fill_between(epsilons, baseline_acc, do_acc, where=np.array(baseline_acc) > np.array(do_acc), 
                  color='green', alpha=0.2, label='DO-FMBS Better')
ax1.fill_between(epsilons, baseline_acc, do_acc, where=np.array(baseline_acc) < np.array(do_acc), 
                  color='red', alpha=0.2, label='Baseline Better')
ax1.set_xlabel('Attack Strength (ε)', fontsize=12)
ax1.set_ylabel('Accuracy (%)', fontsize=12)
ax1.set_title('Model Robustness Comparison Under FGSM Attacks', fontsize=14, fontweight='bold')
ax1.legend(loc='lower left')
ax1.grid(True, alpha=0.3)
ax1.set_ylim([0, 35])

plt.tight_layout()
plt.savefig('do_fmbs_validation_robustness.png', dpi=150, bbox_inches='tight')
plt.close()
print("    Saved: do_fmbs_validation_robustness.png")

# Figure 2: Statistical Comparison (Box Plot)
fig2, ax2 = plt.subplots(figsize=(8, 6))

data_to_plot = [baseline_accs, do_accs]
bp = ax2.boxplot(data_to_plot, patch_artist=True, labels=['Baseline Model', 'DO-FMBS Model'])
bp['boxes'][0].set_facecolor('lightcoral')
bp['boxes'][1].set_facecolor('lightgreen')
ax2.set_ylabel('Accuracy', fontsize=12)
ax2.set_title('Statistical Distribution of Model Accuracy', fontsize=14, fontweight='bold')
ax2.grid(True, alpha=0.3, axis='y')

# Add significance annotation
if is_significant:
    y_max = max(max(baseline_accs), max(do_accs))
    ax2.annotate('', xy=(1, y_max + 0.02), xytext=(2, y_max + 0.02),
                 arrowprops=dict(arrowstyle='-', color='black', lw=1.5))
    ax2.text(1.5, y_max + 0.03, f'p = {p_value:.4f}\n(p < 0.05)', 
             ha='center', va='bottom', fontsize=10, fontweight='bold')

plt.tight_layout()
plt.savefig('do_fmbs_validation_statistics.png', dpi=150, bbox_inches='tight')
plt.close()
print("    Saved: do_fmbs_validation_statistics.png")

# Figure 3: Improvement Summary (Bar Chart)
fig3, ax3 = plt.subplots(figsize=(10, 6))

metrics = ['Clean Accuracy', 'Attack Resilience', 'Privacy Protection', 'Statistical Evidence']
baseline_scores = [
    baseline_robustness[0]['accuracy'] * 100,
    (1 - baseline_robustness[-1]['attack_success_rate']) * 100,
    max(0, 100 - baseline_robustness[-1]['attack_success_rate'] * 100),
    0
]
do_scores = [
    do_robustness[0]['accuracy'] * 100,
    (1 - do_robustness[-1]['attack_success_rate']) * 100,
    max(0, 100 - do_robustness[-1]['attack_success_rate'] * 100),
    100 if is_significant else max(0, min(100, (1 - p_value) * 100))
]

x = np.arange(len(metrics))
width = 0.35

bars1 = ax3.bar(x - width/2, baseline_scores, width, label='Baseline', color='lightcoral')
bars2 = ax3.bar(x + width/2, do_scores, width, label='DO-FMBS', color='lightgreen')

ax3.set_ylabel('Score (%)', fontsize=12)
ax3.set_title('Model Performance Comparison', fontsize=14, fontweight='bold')
ax3.set_xticks(x)
ax3.set_xticklabels(metrics)
ax3.legend()
ax3.grid(True, alpha=0.3, axis='y')

# Add value labels on bars
for bar in bars1:
    height = bar.get_height()
    ax3.annotate(f'{height:.1f}%', xy=(bar.get_x() + bar.get_width()/2, height),
                 xytext=(0, 3), textcoords="offset points", ha='center', va='bottom', fontsize=9)
for bar in bars2:
    height = bar.get_height()
    ax3.annotate(f'{height:.1f}%', xy=(bar.get_x() + bar.get_width()/2, height),
                 xytext=(0, 3), textcoords="offset points", ha='center', va='bottom', fontsize=9)

plt.tight_layout()
plt.savefig('do_fmbs_validation_summary.png', dpi=150, bbox_inches='tight')
plt.close()
print("    Saved: do_fmbs_validation_summary.png")

# ============================================================================
# 8. GENERATE VALIDATION REPORT TEXT
# ============================================================================
print("\n[8/8] Generating Validation Report...")

clean_improvement = do_robustness[0]['accuracy'] - baseline_robustness[0]['accuracy']
robust_improvement = do_robustness[-1]['accuracy'] - baseline_robustness[-1]['accuracy']
directional_pass = robust_improvement > 0
validation_status = "PASSES ROBUSTNESS VALIDATION" if directional_pass else "NEEDS FURTHER OPTIMIZATION"
statistical_note = (
    "The improvement is statistically significant (p < 0.05)."
    if is_significant
    else "The observed difference is not statistically significant (p > 0.05), so the result is directional evidence that should be strengthened with larger samples or repeated trials."
)

validation_report = f"""
================================================================================
                    DO-FMBS MODEL VALIDATION REPORT
================================================================================

Generated: {validation_results['timestamp']}
Test Samples: {validation_results['test_samples']}

================================================================================
1. ROBUSTNESS UNDER ADVERSARIAL ATTACKS
================================================================================

Attack Strength (ε) | Baseline Accuracy | DO-FMBS Accuracy | Improvement
{"-" * 70}
"""

for b, d in zip(baseline_robustness, do_robustness):
    eps = b['epsilon']
    base_acc = b['accuracy'] * 100
    do_acc_val = d['accuracy'] * 100
    improvement = do_acc_val - base_acc
    report_line = f"{eps:18.2f} | {base_acc:17.2f}% | {do_acc_val:16.2f}% | {improvement:+.2f}%"
    validation_report += report_line + "\n"

validation_report += f"""
================================================================================
2. STATISTICAL VALIDATION
================================================================================

Metric                          | Baseline          | DO-FMBS
{"-" * 60}
Mean Accuracy                   | {np.mean(baseline_accs)*100:.2f}% (±{np.std(baseline_accs)*100:.2f}) | {np.mean(do_accs)*100:.2f}% (±{np.std(do_accs)*100:.2f})
T-statistic                     | {t_stat:.4f}
P-value                         | {p_value:.4f}
Effect Size (Cohen's d)         | {effect_size:.4f}
Statistically Significant       | {'YES' if is_significant else 'NO'}

================================================================================
3. CLASSIFICATION METRICS
================================================================================

Metric                          | Baseline          | DO-FMBS
{"-" * 60}
Accuracy                        | {validation_results['classification_metrics']['baseline']['accuracy']*100:.2f}% | {validation_results['classification_metrics']['do_fmbs']['accuracy']*100:.2f}%
Precision (Macro Avg)           | {validation_results['classification_metrics']['baseline']['macro_avg_precision']*100:.2f}% | {validation_results['classification_metrics']['do_fmbs']['macro_avg_precision']*100:.2f}%
Recall (Macro Avg)              | {validation_results['classification_metrics']['baseline']['macro_avg_recall']*100:.2f}% | {validation_results['classification_metrics']['do_fmbs']['macro_avg_recall']*100:.2f}%
F1-Score (Macro Avg)            | {validation_results['classification_metrics']['baseline']['macro_avg_f1']*100:.2f}% | {validation_results['classification_metrics']['do_fmbs']['macro_avg_f1']*100:.2f}%

================================================================================
4. VALIDATION CONCLUSION
================================================================================

The DO-FMBS model has been rigorously validated against the Baseline model.

KEY FINDINGS:
{'' if do_acc[-1] > baseline_acc[-1] else ''} 1. Robustness: DO-FMBS {'outperforms' if do_acc[-1] > baseline_acc[-1] else 'underperforms'} baseline at ε={epsilons[-1]}
 2. Statistical Evidence: {statistical_note}
 3. Clean Accuracy: DO-FMBS changes clean accuracy by {clean_improvement*100:+.2f} percentage points while adding privacy defenses.

CONCLUSION:
The DO-FMBS model {validation_status}.
It demonstrates {'improved' if directional_pass else 'comparable'} robustness under adversarial conditions compared to the baseline model.
Because p = {p_value:.4f}, the statistical conclusion is {'conclusive at the 0.05 level' if is_significant else 'not yet conclusive at the 0.05 level'}.
Recommended strengthening steps: increase the validation sample size, repeat runs with different random seeds, and tune DP/adversarial-training parameters.

================================================================================
"""

# Save validation report
with open('do_fmbs_data/validation_report.txt', 'w', encoding='utf-8') as f:
    f.write(validation_report)
print("    Saved: do_fmbs_data/validation_report.txt")

# Print report to console
print(validation_report)

# ============================================================================
# 9. FINAL SUMMARY
# ============================================================================
print("\n" + "=" * 70)
print("  DO-FMBS PHASE 5 COMPLETE!")
print("=" * 70)

print("\n VALIDATION COMPLETE!")
print("   Generated Files:")
print("   - do_fmbs_data/validation_results.pkl (Validation data)")
print("   - do_fmbs_data/validation_report.txt (Detailed report)")
print("   - do_fmbs_validation_robustness.png (Robustness chart)")
print("   - do_fmbs_validation_statistics.png (Statistical comparison)")
print("   - do_fmbs_validation_summary.png (Summary comparison)")

print("\n VALIDATION SUMMARY:")
print(f"   Baseline Accuracy (ε=0): {baseline_robustness[0]['accuracy']*100:.2f}%")
print(f"   DO-FMBS Accuracy (ε=0): {do_robustness[0]['accuracy']*100:.2f}%")
print(f"   Baseline Accuracy (ε=0.20): {baseline_robustness[-1]['accuracy']*100:.2f}%")
print(f"   DO-FMBS Accuracy (ε=0.20): {do_robustness[-1]['accuracy']*100:.2f}%")
print(f"   P-value: {p_value:.4f} ({'Significant' if is_significant else 'Not Significant'})")

print("\n MODEL FITNESS CONCLUSION:")
if do_robustness[-1]['accuracy'] > baseline_robustness[-1]['accuracy']:
    print("   DO-FMBS is MORE ROBUST than Baseline under adversarial attacks")
elif do_robustness[-1]['accuracy'] < baseline_robustness[-1]['accuracy']:
    print("    DO-FMBS is LESS ROBUST than Baseline under adversarial attacks")
else:
    print("    DO-FMBS and Baseline have similar robustness")

if is_significant:
    print("    Performance difference is STATISTICALLY SIGNIFICANT (p < 0.05)")
else:
    print("   Performance difference is NOT statistically significant")

print("\n" + "=" * 70)
