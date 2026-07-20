"""
DO-FMBS Research Project - Phase 3: Adversarial Attack Simulation
Privacy-Preservation Model for Federated Multimodal Biometric System

Researcher: Lydiah Moraa Machora | Reg No: I152/P/9515/23
University: Jaramogi Oginga Odinga University of Science and Technology

This script:
1. Loads the baseline model from Phase 2
2. Generates adversarial examples (FGSM, PGD)
3. Evaluates model vulnerability
4. Measures attack success rates
"""

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import os
import pickle
import argparse

parser = argparse.ArgumentParser(description="Run Phase 3 attack simulations.")
parser.add_argument("--epsilon", type=float, default=0.10)
parser.add_argument("--membership-threshold", type=float, default=0.70)
parser.add_argument("--poison-rate", type=float, default=0.10)
parser.add_argument("--poison-epochs", type=int, default=3)
args = parser.parse_args()

# Suppress warnings including absl
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # 0=all, 1=info, 2=warning, 3=error
import logging
logging.getLogger('tensorflow').setLevel(logging.ERROR)
logging.getLogger('absl').setLevel(logging.ERROR)

# Set random seeds
tf.random.set_seed(42)
np.random.seed(42)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

print("=" * 60)
print("DO-FMBS Phase 3: Adversarial Attack Simulation")
print("=" * 60)

# ============================================================================
# 1. LOAD THE BASELINE MODEL
# ============================================================================
print("\n[1/5] Loading Baseline Model...")

# Load face test data
face_data = np.load('do_fmbs_data/face_data.npz', allow_pickle=True)
X_test_face = face_data['X_test']
y_test_face = face_data['y_test']
X_train_face = face_data['X_train']
y_train_face = face_data['y_train']
n_face_features = int(face_data['n_features'])
n_face_classes = int(face_data['n_classes'])

# Load fingerprint test data
finger_data = np.load('do_fmbs_data/fingerprint_data.npz', allow_pickle=True)
X_test_finger = finger_data['X_test']
y_test_finger = finger_data['y_test']
X_train_finger = finger_data['X_train']
y_train_finger = finger_data['y_train']
n_finger_features = int(finger_data['n_features'])
n_finger_classes = int(finger_data['n_classes'])

# Load the model
model = tf.keras.models.load_model('do_fmbs_data/baseline_model.keras')
print("    Model loaded successfully")

# Prepare test data (matching pairs)
n_samples = min(len(X_test_face), len(X_test_finger), 100)
X_face_test = X_test_face[:n_samples].astype(np.float32)
X_finger_test = X_test_finger[:n_samples].astype(np.float32)
y_face_test = y_test_face[:n_samples]
y_finger_test = y_test_finger[:n_samples]

n_train_attack = min(len(X_train_face), len(X_train_finger), 200)
X_face_train_attack = X_train_face[:n_train_attack].astype(np.float32)
X_finger_train_attack = X_train_finger[:n_train_attack].astype(np.float32)
y_face_train_attack = y_train_face[:n_train_attack]
y_finger_train_attack = y_train_finger[:n_train_attack]

print(f"    Test samples: {n_samples}")
print(f"    Face features: {n_face_features}")
print(f"    Fingerprint features: {n_finger_features}")

# ============================================================================
# 2. BASELINE ACCURACY (BEFORE ATTACK)
# ============================================================================
print("\n[2/5] Baseline Accuracy (Before Attacks)...")

baseline_results = model.evaluate(
    x={'face_input': X_face_test, 'finger_input': X_finger_test},
    y={'face_output': y_face_test, 'finger_output': y_finger_test},
    verbose=0
)

baseline_face_acc = baseline_results[3]
baseline_finger_acc = baseline_results[4]

print(f"    Face Accuracy: {baseline_face_acc:.4f} ({baseline_face_acc*100:.2f}%)")
print(f"    Fingerprint Accuracy: {baseline_finger_acc:.4f} ({baseline_finger_acc*100:.2f}%)")

# ============================================================================
# 3. IMPLEMENT ADVERSARIAL ATTACKS
# ============================================================================
print("\n[3/5] Implementing Adversarial Attacks...")

def fgsm_attack(model, face_input, finger_input, face_label, finger_label, epsilon=0.1):
    """
    Fast Gradient Sign Method (FGSM) Attack
    Perturbs input in the direction of the gradient to increase loss
    """
    # Use only face modality for attack (simpler)
    face_input_tensor = tf.convert_to_tensor(face_input.reshape(1, -1))
    
    with tf.GradientTape() as tape:
        tape.watch(face_input_tensor)
        predictions = model({'face_input': face_input_tensor, 
                            'finger_input': finger_input.reshape(1, -1)})
        face_pred = predictions[0]
        loss = tf.keras.losses.sparse_categorical_crossentropy([face_label], face_pred)
    
    # Get gradients
    gradients = tape.gradient(loss, face_input_tensor)
    
    # Create perturbation
    perturbation = epsilon * tf.sign(gradients)
    
    # Generate adversarial example
    adversarial_face = face_input_tensor + perturbation
    adversarial_face = tf.clip_by_value(adversarial_face, -3, 3)
    
    return adversarial_face.numpy().reshape(-1)

def pgd_attack(model, face_input, finger_input, face_label, finger_label, 
               epsilon=0.1, alpha=0.01, iterations=10):
    """
    Projected Gradient Descent (PGD) Attack
    Iterative version of FGSM - stronger attack
    """
    adversarial_face = face_input.copy()
    
    for _ in range(iterations):
        adversarial_tensor = tf.convert_to_tensor(adversarial_face.reshape(1, -1))
        
        with tf.GradientTape() as tape:
            tape.watch(adversarial_tensor)
            predictions = model({'face_input': adversarial_tensor,
                                'finger_input': finger_input.reshape(1, -1)})
            face_pred = predictions[0]
            loss = tf.keras.losses.sparse_categorical_crossentropy([face_label], face_pred)
        
        gradients = tape.gradient(loss, adversarial_tensor)
        perturbation = alpha * tf.sign(gradients)
        adversarial_face = adversarial_face + perturbation.numpy().reshape(-1)
        
        # Project back to epsilon ball
        perturbation = adversarial_face - face_input
        perturbation = np.clip(perturbation, -epsilon, epsilon)
        adversarial_face = face_input + perturbation
        adversarial_face = np.clip(adversarial_face, -3, 3)
    
    return adversarial_face

def evaluate_attack(model, X_face, X_finger, y_face, y_finger, 
                    attack_type='fgsm', epsilon=0.1):
    """
    Evaluate model under adversarial attack
    """
    correct_before = 0
    correct_after = 0
    success_count = 0
    
    for i in range(len(X_face)):
        # Check original prediction
        orig_pred = model.predict(
            {'face_input': X_face[i:i+1], 'finger_input': X_finger[i:i+1]},
            verbose=0
        )
        orig_face_pred = np.argmax(orig_pred[0])
        
        if orig_face_pred == y_face[i]:
            correct_before += 1
            
            # Generate adversarial example
            if attack_type == 'fgsm':
                adv_face = fgsm_attack(model, X_face[i], X_finger[i], 
                                        y_face[i], y_finger[i], epsilon)
            else:  # pgd
                adv_face = pgd_attack(model, X_face[i], X_finger[i],
                                       y_face[i], y_finger[i], epsilon)
            
            # Test adversarial example
            adv_pred = model.predict(
                {'face_input': adv_face.reshape(1, -1), 
                 'finger_input': X_finger[i:i+1]},
                verbose=0
            )
            adv_face_pred = np.argmax(adv_pred[0])
            
            if adv_face_pred != y_face[i]:
                correct_after += 1
                success_count += 1
    
    attack_success_rate = success_count / correct_before if correct_before > 0 else 0
    
    return {
        'correct_before': correct_before,
        'correct_after': correct_after,
        'success_count': success_count,
        'attack_success_rate': attack_success_rate
    }

print("    Attack functions defined")

def membership_inference_attack(model, X_member_face, X_member_finger, X_nonmember_face,
                                X_nonmember_finger, threshold=0.70):
    """
    Confidence-threshold membership inference attack.
    Samples with max prediction confidence above the threshold are inferred as
    training members.
    """
    member_pred = model.predict(
        {'face_input': X_member_face, 'finger_input': X_member_finger},
        verbose=0
    )[0]
    nonmember_pred = model.predict(
        {'face_input': X_nonmember_face, 'finger_input': X_nonmember_finger},
        verbose=0
    )[0]

    member_conf = np.max(member_pred, axis=1)
    nonmember_conf = np.max(nonmember_pred, axis=1)

    true_labels = np.array([1] * len(member_conf) + [0] * len(nonmember_conf))
    inferred = np.array(
        list(member_conf >= threshold) + list(nonmember_conf >= threshold),
        dtype=int
    )

    attack_accuracy = np.mean(inferred == true_labels)
    true_member_detection = np.mean(member_conf >= threshold)
    false_member_rate = np.mean(nonmember_conf >= threshold)
    privacy_risk = max(0.0, attack_accuracy - 0.5) * 2

    return {
        'attack': 'Membership Inference Attack',
        'type': 'Inference Attack',
        'threshold': float(threshold),
        'attack_accuracy': float(attack_accuracy),
        'true_member_detection_rate': float(true_member_detection),
        'false_member_rate': float(false_member_rate),
        'privacy_risk_score': float(privacy_risk),
        'member_mean_confidence': float(np.mean(member_conf)),
        'nonmember_mean_confidence': float(np.mean(nonmember_conf))
    }

def model_poisoning_attack(model, X_face, X_finger, y_face, y_finger,
                           X_eval_face, X_eval_finger, y_eval_face,
                           poison_rate=0.10, epochs=3):
    """
    Model poisoning attack using label flipping on a subset of local training data.
    The poisoned model is fine-tuned from the baseline and compared against the
    clean baseline on the same evaluation split.
    """
    poison_rate = float(np.clip(poison_rate, 0.0, 0.5))
    poisoned_model = tf.keras.models.clone_model(model)
    poisoned_model.set_weights(model.get_weights())
    poisoned_model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.0005),
        loss={'face_output': 'sparse_categorical_crossentropy',
              'finger_output': 'sparse_categorical_crossentropy'},
        metrics={'face_output': ['accuracy'], 'finger_output': ['accuracy']}
    )

    clean_pred = model.predict(
        {'face_input': X_eval_face, 'finger_input': X_eval_finger},
        verbose=0
    )[0]
    clean_accuracy = float(np.mean(np.argmax(clean_pred, axis=1) == y_eval_face))

    y_face_poisoned = np.array(y_face, copy=True)
    y_finger_poisoned = np.array(y_finger, copy=True)
    n_poison = max(1, int(len(y_face_poisoned) * poison_rate))
    poison_indices = np.random.choice(len(y_face_poisoned), size=n_poison, replace=False)
    y_face_poisoned[poison_indices] = (y_face_poisoned[poison_indices] + 1) % n_face_classes
    y_finger_poisoned[poison_indices] = (y_finger_poisoned[poison_indices] + 1) % n_finger_classes

    poisoned_model.fit(
        x={'face_input': X_face, 'finger_input': X_finger},
        y={'face_output': y_face_poisoned, 'finger_output': y_finger_poisoned},
        epochs=epochs,
        batch_size=32,
        verbose=0
    )

    poisoned_pred = poisoned_model.predict(
        {'face_input': X_eval_face, 'finger_input': X_eval_finger},
        verbose=0
    )[0]
    poisoned_accuracy = float(np.mean(np.argmax(poisoned_pred, axis=1) == y_eval_face))

    return {
        'attack': 'Label-Flipping Model Poisoning',
        'type': 'Poisoning Attack',
        'poison_rate': poison_rate,
        'poisoned_samples': int(n_poison),
        'clean_accuracy': clean_accuracy,
        'poisoned_accuracy': poisoned_accuracy,
        'accuracy_drop': float(clean_accuracy - poisoned_accuracy),
        'attack_success_rate': float(max(0.0, clean_accuracy - poisoned_accuracy))
    }

# ============================================================================
# 4. RUN ATTACKS WITH DIFFERENT EPSILON VALUES
# ============================================================================
print("\n[4/5] Running Adversarial Attacks...")

epsilons = sorted(set([0.01, 0.05, 0.1, 0.2, 0.3, round(args.epsilon, 2)]))
fgsm_results = []
pgd_results = []

print("\n    FGSM Attack Results:")
print("    " + "-" * 50)
print("    Epsilon | Attack Success Rate")
print("    " + "-" * 50)

for eps in epsilons:
    result = evaluate_attack(model, X_face_test, X_finger_test, 
                              y_face_test, y_finger_test, 
                              attack_type='fgsm', epsilon=eps)
    fgsm_results.append({'epsilon': eps, 'success_rate': result['attack_success_rate']})
    print(f"    {eps:7.2f} | {result['attack_success_rate']*100:6.2f}%")

print("\n    PGD Attack Results (10 iterations):")
print("    " + "-" * 50)
print("    Epsilon | Attack Success Rate")
print("    " + "-" * 50)

for eps in epsilons:
    result = evaluate_attack(model, X_face_test, X_finger_test,
                              y_face_test, y_finger_test,
                              attack_type='pgd', epsilon=eps)
    pgd_results.append({'epsilon': eps, 'success_rate': result['attack_success_rate']})
    print(f"    {eps:7.2f} | {result['attack_success_rate']*100:6.2f}%")

print("\n    Membership Inference Attack Results:")
membership_results = membership_inference_attack(
    model,
    X_face_train_attack[:n_samples], X_finger_train_attack[:n_samples],
    X_face_test, X_finger_test,
    threshold=args.membership_threshold
)
print(f"    Attack Accuracy: {membership_results['attack_accuracy']*100:.2f}%")
print(f"    Privacy Risk Score: {membership_results['privacy_risk_score']*100:.2f}%")

print("\n    Model Poisoning Attack Results:")
poisoning_results = model_poisoning_attack(
    model,
    X_face_train_attack, X_finger_train_attack, y_face_train_attack, y_finger_train_attack,
    X_face_test, X_finger_test, y_face_test,
    poison_rate=args.poison_rate,
    epochs=args.poison_epochs
)
print(f"    Poison Rate: {poisoning_results['poison_rate']*100:.1f}%")
print(f"    Accuracy Drop: {poisoning_results['accuracy_drop']*100:.2f}%")

# ============================================================================
# 5. VISUALIZE ATTACK RESULTS
# ============================================================================
print("\n[5/5] Visualizing Results...")

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Plot FGSM results
eps_values = [r['epsilon'] for r in fgsm_results]
success_rates = [r['success_rate'] * 100 for r in fgsm_results]

axes[0].plot(eps_values, success_rates, 'b-o', linewidth=2, markersize=8)
axes[0].set_xlabel('Epsilon (Attack Strength)', fontsize=12)
axes[0].set_ylabel('Attack Success Rate (%)', fontsize=12)
axes[0].set_title('FGSM Attack: Face Recognition', fontsize=12, fontweight='bold')
axes[0].grid(True, alpha=0.3)
axes[0].set_ylim([0, 100])

# Plot PGD results
eps_values = [r['epsilon'] for r in pgd_results]
success_rates = [r['success_rate'] * 100 for r in pgd_results]

axes[1].plot(eps_values, success_rates, 'r-o', linewidth=2, markersize=8)
axes[1].set_xlabel('Epsilon (Attack Strength)', fontsize=12)
axes[1].set_ylabel('Attack Success Rate (%)', fontsize=12)
axes[1].set_title('PGD Attack: Face Recognition', fontsize=12, fontweight='bold')
axes[1].grid(True, alpha=0.3)
axes[1].set_ylim([0, 100])

plt.tight_layout()
plt.savefig('do_fmbs_attack_results.png', dpi=120, bbox_inches='tight')
plt.close()
print("    Saved: do_fmbs_attack_results.png")

fig_ext, axes_ext = plt.subplots(1, 2, figsize=(14, 5))

inference_labels = ['Attack Accuracy', 'Privacy Risk', 'Member Detection', 'False Member Rate']
inference_values = [
    membership_results['attack_accuracy'] * 100,
    membership_results['privacy_risk_score'] * 100,
    membership_results['true_member_detection_rate'] * 100,
    membership_results['false_member_rate'] * 100
]
axes_ext[0].bar(inference_labels, inference_values, color=['#2b6cb0', '#805ad5', '#38a169', '#dd6b20'])
axes_ext[0].set_ylabel('Rate (%)')
axes_ext[0].set_title('Membership Inference Attack', fontsize=12, fontweight='bold')
axes_ext[0].set_ylim([0, 100])
axes_ext[0].tick_params(axis='x', rotation=20)
axes_ext[0].grid(True, axis='y', alpha=0.3)

poison_labels = ['Clean Accuracy', 'Poisoned Accuracy', 'Accuracy Drop']
poison_values = [
    poisoning_results['clean_accuracy'] * 100,
    poisoning_results['poisoned_accuracy'] * 100,
    poisoning_results['accuracy_drop'] * 100
]
axes_ext[1].bar(poison_labels, poison_values, color=['#38a169', '#e53e3e', '#dd6b20'])
axes_ext[1].set_ylabel('Accuracy / Drop (%)')
axes_ext[1].set_title('Label-Flipping Model Poisoning', fontsize=12, fontweight='bold')
axes_ext[1].set_ylim([0, max(10, max(poison_values) + 5)])
axes_ext[1].tick_params(axis='x', rotation=15)
axes_ext[1].grid(True, axis='y', alpha=0.3)

plt.tight_layout()
plt.savefig('do_fmbs_attack_extended_results.png', dpi=120, bbox_inches='tight')
plt.close()
print("    Saved: do_fmbs_attack_extended_results.png")

# ============================================================================
# 6. VISUALIZE ADVERSARIAL EXAMPLES
# ============================================================================
print("\n[6/5] Visualizing Adversarial Examples...")

# Select a sample
sample_idx = 0
sample_face = X_face_test[sample_idx].reshape(1, -1)
sample_finger = X_finger_test[sample_idx].reshape(1, -1)
sample_face_label = y_face_test[sample_idx]

# Generate adversarial example
adversarial_face = fgsm_attack(model, X_face_test[sample_idx], 
                                X_finger_test[sample_idx],
                                y_face_test[sample_idx], 
                                y_finger_test[sample_idx], 
                                epsilon=0.1)

# Get predictions
orig_pred = model.predict({'face_input': sample_face, 'finger_input': sample_finger}, verbose=0)
adv_pred = model.predict({'face_input': adversarial_face.reshape(1, -1), 
                          'finger_input': sample_finger}, verbose=0)

orig_class = np.argmax(orig_pred[0])
adv_class = np.argmax(adv_pred[0])

print(f"\n    Sample #{sample_idx}")
print(f"    Original face class: {sample_face_label}")
print(f"    Predicted class (original): {orig_class}")
print(f"    Predicted class (adversarial): {adv_class}")

# Visualize original vs adversarial (if we had image shape info)
# Note: We're working with flattened features, so we can't easily visualize
print("\n    Note: Adversarial examples are in feature space (flattened)")
print("    In production, these would be imperceptible perturbations to images")

# ============================================================================
# 7. SAVE RESULTS
# ============================================================================
print("\n[7/5] Saving Attack Results...")

attack_results = {
    'baseline_face_accuracy': float(baseline_face_acc),
    'baseline_finger_accuracy': float(baseline_finger_acc),
    'fgsm_results': fgsm_results,
    'pgd_results': pgd_results,
    'membership_inference': membership_results,
    'model_poisoning': poisoning_results,
    'epsilons': epsilons,
    'attack_manifest': [
        {'type': 'Evasion Attack', 'attacks': ['FGSM', 'PGD'], 'status': 'Applied'},
        {'type': 'Inference Attack', 'attacks': ['Membership Inference Attack'], 'status': 'Applied'},
        {'type': 'Poisoning Attack', 'attacks': ['Label-Flipping Model Poisoning'], 'status': 'Applied'}
    ],
    'parameters': {
        'selected_epsilon': args.epsilon,
        'membership_threshold': args.membership_threshold,
        'poison_rate': args.poison_rate,
        'poison_epochs': args.poison_epochs
    }
}

with open('do_fmbs_data/attack_results.pkl', 'wb') as f:
    pickle.dump(attack_results, f)
print("    Saved: do_fmbs_data/attack_results.pkl")

# ============================================================================
# 8. FINAL SUMMARY
# ============================================================================
print("\n" + "=" * 60)
print("  DO-FMBS PHASE 3 COMPLETE!")
print("=" * 60)

print("\n ATTACK RESULTS SUMMARY:")
print(f"  Baseline Face Accuracy: {baseline_face_acc:.2%}")
print(f"\n  FGSM Attack (epsilon=0.1):")
fgsm_eps01 = next(r for r in fgsm_results if r['epsilon'] == 0.1)
print(f"    Attack Success Rate: {fgsm_eps01['success_rate']*100:.2f}%")
print(f"\n  PGD Attack (epsilon=0.1, 10 iterations):")
pgd_eps01 = next(r for r in pgd_results if r['epsilon'] == 0.1)
print(f"    Attack Success Rate: {pgd_eps01['success_rate']*100:.2f}%")
print(f"\n  Membership Inference:")
print(f"    Attack Accuracy: {membership_results['attack_accuracy']*100:.2f}%")
print(f"\n  Model Poisoning:")
print(f"    Accuracy Drop: {poisoning_results['accuracy_drop']*100:.2f}%")

print("\n SAVED FILES:")
print("  - do_fmbs_data/attack_results.pkl")
print("  - do_fmbs_attack_results.png")

print("\n NEXT STEP: Phase 4 - Defense Mechanisms")
print("   (Differential Privacy + Adversarial Training)")
print("=" * 60)
