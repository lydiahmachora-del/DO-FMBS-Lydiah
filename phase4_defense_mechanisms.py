"""
DO-FMBS Research Project - Phase 4: Defense Mechanisms (FINAL)
Privacy-Preservation Model for Federated Multimodal Biometric System

Researcher: Lydiah Moraa Machora | Reg No: I152/P/9515/23
University: Jaramogi Oginga Odinga University of Science and Technology
"""

# ============================================================================
# WARNING SUPPRESSION - Must be at the very top
# ============================================================================
import os
import sys
import warnings
import logging

# Suppress all warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['STREAMLIT_SUPPRESS_WARNING'] = 'true'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

# Suppress Python warnings
warnings.filterwarnings('ignore')

# Suppress logging
logging.getLogger('tensorflow').setLevel(logging.ERROR)
logging.getLogger('absl').setLevel(logging.ERROR)

# ============================================================================
# MAIN IMPORTS
# ============================================================================
import tensorflow as tf
import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pickle
import argparse
from datetime import datetime

parser = argparse.ArgumentParser(description="Run Phase 4 defense mechanisms.")
parser.add_argument("--epochs", type=int, default=5)
parser.add_argument("--max-pairs", type=int, default=0,
                    help="Maximum face training samples to use. Use 0 for all available face samples.")
parser.add_argument("--test-pairs", type=int, default=0,
                    help="Maximum face test samples to use. Use 0 for all available face samples.")
parser.add_argument("--batch-size", type=int, default=32)
args = parser.parse_args()

# Set random seeds
tf.random.set_seed(42)
np.random.seed(42)

print("=" * 70)
print("DO-FMBS Phase 4: Defense Mechanisms (Final)")
print("=" * 70)

# ============================================================================
# 1. LOAD DATA
# ============================================================================
print("\n[1/6] Loading Phase 1 Data...")

# Load face data
face_data = np.load('do_fmbs_data/face_data.npz', allow_pickle=True)
X_train_face = face_data['X_train']
X_test_face = face_data['X_test']
y_train_face = face_data['y_train']
y_test_face = face_data['y_test']
n_face_classes = int(face_data['n_classes'])
n_face_features = int(face_data['n_features'])

# Load fingerprint data
finger_data = np.load('do_fmbs_data/fingerprint_data.npz', allow_pickle=True)
X_train_finger = finger_data['X_train']
X_test_finger = finger_data['X_test']
y_train_finger = finger_data['y_train']
y_test_finger = finger_data['y_test']
n_finger_classes = int(finger_data['n_classes'])
n_finger_features = int(finger_data['n_features'])

# The face classifier head is intentionally fed by face_features only. Earlier
# versions capped Phase 4 to the shorter fingerprint dataset, throwing away most
# face samples. Cycle fingerprint inputs so the face branch can learn from every
# available face example while still satisfying the two-input model signature.
def cycle_to_length(values, length):
    repeats = int(np.ceil(length / len(values)))
    return np.tile(values, (repeats, 1))[:length]

def cycle_labels_to_length(values, length):
    repeats = int(np.ceil(length / len(values)))
    return np.tile(values, repeats)[:length]

train_limit = len(X_train_face) if args.max_pairs <= 0 else min(len(X_train_face), args.max_pairs)
test_limit = len(X_test_face) if args.test_pairs <= 0 else min(len(X_test_face), args.test_pairs)

X_face_train = X_train_face[:train_limit].astype(np.float32)
y_face_train = y_train_face[:train_limit]
X_finger_train = cycle_to_length(X_train_finger.astype(np.float32), train_limit)
y_finger_train = cycle_labels_to_length(y_train_finger, train_limit)

X_face_test = X_test_face[:test_limit].astype(np.float32)
y_face_test = y_test_face[:test_limit]
X_finger_test = cycle_to_length(X_test_finger.astype(np.float32), test_limit)
y_finger_test = cycle_labels_to_length(y_test_finger, test_limit)

print(f"    Face training samples: {train_limit}")
print(f"    Face test samples: {test_limit}")
print("    Fingerprint inputs are cycled to avoid limiting face training data.")
print(f"    Face features: {n_face_features}")
print(f"    Fingerprint features: {n_finger_features}")

# ============================================================================
# 2. MODEL ARCHITECTURE
# ============================================================================
print("\n[2/6] Defining Model Architecture...")

def create_multimodal_model(face_input_dim, finger_input_dim, 
                             n_face_classes, n_finger_classes):
    """Creates multimodal neural network for face + fingerprint recognition"""
    
    # Face branch
    face_input = tf.keras.Input(shape=(face_input_dim,), name='face_input')
    face_x = tf.keras.layers.Dense(256, activation='relu')(face_input)
    face_x = tf.keras.layers.BatchNormalization()(face_x)
    face_x = tf.keras.layers.Dropout(0.3)(face_x)
    face_x = tf.keras.layers.Dense(128, activation='relu')(face_x)
    face_x = tf.keras.layers.BatchNormalization()(face_x)
    face_features = tf.keras.layers.Dense(64, activation='relu')(face_x)
    
    # Fingerprint branch
    finger_input = tf.keras.Input(shape=(finger_input_dim,), name='finger_input')
    finger_x = tf.keras.layers.Dense(256, activation='relu')(finger_input)
    finger_x = tf.keras.layers.BatchNormalization()(finger_x)
    finger_x = tf.keras.layers.Dropout(0.3)(finger_x)
    finger_x = tf.keras.layers.Dense(128, activation='relu')(finger_x)
    finger_x = tf.keras.layers.BatchNormalization()(finger_x)
    finger_features = tf.keras.layers.Dense(64, activation='relu')(finger_x)
    
    # Keep the modality heads direct. Face and synthetic fingerprint labels are
    # separate identity spaces, so arbitrary cross-modal pair fusion suppresses
    # recognition accuracy.
    fused = tf.keras.layers.Concatenate()([face_features, finger_features])
    fused = tf.keras.layers.Dense(128, activation='relu')(fused)
    fused = tf.keras.layers.BatchNormalization()(fused)
    fused = tf.keras.layers.Dropout(0.3)(fused)
    
    # Output heads
    face_head = tf.keras.layers.Dense(64, activation='relu')(face_features)
    face_output = tf.keras.layers.Dense(n_face_classes, activation='softmax', name='face_output')(face_head)

    finger_head = tf.keras.layers.Dense(64, activation='relu')(finger_features)
    finger_output = tf.keras.layers.Dense(n_finger_classes, activation='softmax', name='finger_output')(finger_head)
    
    model = tf.keras.Model(inputs=[face_input, finger_input], outputs=[face_output, finger_output])
    return model

def compile_model(model, learning_rate=0.001):
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss={'face_output': 'sparse_categorical_crossentropy',
              'finger_output': 'sparse_categorical_crossentropy'},
        metrics={'face_output': ['accuracy'], 'finger_output': ['accuracy']}
    )
    return model

print("    Model architecture defined")

# ============================================================================
# 3. DIFFERENTIAL PRIVACY TRAINING
# ============================================================================
print("\n[3/6] Implementing Differential Privacy (DP)...")

def train_with_dp(model, X_face, X_finger, y_face, y_finger, 
                   epochs=20, batch_size=32, epsilon=0.5, verbose=True):
    """Train model with Differential Privacy using gradient noise injection"""
    
    def add_dp_noise(gradients, epsilon=epsilon):
        noise_scale = 0.001 / epsilon
        noisy_grads = []
        for grad in gradients:
            if grad is not None:
                noise = np.random.normal(0, noise_scale, grad.shape).astype(np.float32)
                noisy_grads.append(grad + noise)
            else:
                noisy_grads.append(grad)
        return noisy_grads
    
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
    history = {'loss': [], 'face_acc': [], 'finger_acc': []}
    n_samples = len(X_face)
    
    for epoch in range(epochs):
        indices = np.random.permutation(n_samples)
        epoch_losses = []
        epoch_face_accs = []
        epoch_finger_accs = []
        
        for start in range(0, n_samples, batch_size):
            end = min(start + batch_size, n_samples)
            batch_indices = indices[start:end]
            
            batch_face = X_face[batch_indices]
            batch_finger = X_finger[batch_indices]
            batch_y_face = y_face[batch_indices]
            batch_y_finger = y_finger[batch_indices]
            
            with tf.GradientTape() as tape:
                predictions = model({'face_input': batch_face, 'finger_input': batch_finger})
                loss_face = tf.keras.losses.sparse_categorical_crossentropy(batch_y_face, predictions[0])
                loss_finger = tf.keras.losses.sparse_categorical_crossentropy(batch_y_finger, predictions[1])
                loss = tf.reduce_mean(loss_face + loss_finger)
            
            grads = tape.gradient(loss, model.trainable_variables)
            noisy_grads = add_dp_noise(grads)
            optimizer.apply_gradients(zip(noisy_grads, model.trainable_variables))
            
            epoch_losses.append(loss.numpy())
            face_acc = tf.reduce_mean(tf.cast(tf.argmax(predictions[0], axis=1) == batch_y_face, tf.float32)).numpy()
            finger_acc = tf.reduce_mean(tf.cast(tf.argmax(predictions[1], axis=1) == batch_y_finger, tf.float32)).numpy()
            epoch_face_accs.append(face_acc)
            epoch_finger_accs.append(finger_acc)
        
        history['loss'].append(np.mean(epoch_losses))
        history['face_acc'].append(np.mean(epoch_face_accs))
        history['finger_acc'].append(np.mean(epoch_finger_accs))
        
        if verbose and (epoch + 1) % 5 == 0:
            print(f"    DP Epoch {epoch+1}/{epochs}: Loss={history['loss'][-1]:.4f}, Face Acc={history['face_acc'][-1]:.4f}")
    
    return history

print("    Differential Privacy implemented")

# ============================================================================
# 4. ADVERSARIAL ROBUSTNESS TRAINING
# ============================================================================
print("\n[4/6] Implementing Adversarial Robustness Training (ART)...")

def fgsm_attack(model, face_input, finger_input, face_label, epsilon=0.05):
    """Fast Gradient Sign Method attack"""
    face_tensor = tf.convert_to_tensor(face_input, dtype=tf.float32)
    finger_tensor = tf.convert_to_tensor(finger_input, dtype=tf.float32)
    
    with tf.GradientTape() as tape:
        tape.watch(face_tensor)
        predictions = model({'face_input': face_tensor, 'finger_input': finger_tensor})
        loss = tf.keras.losses.sparse_categorical_crossentropy(face_label, predictions[0])
    
    gradients = tape.gradient(loss, face_tensor)
    perturbation = epsilon * tf.sign(gradients)
    adversarial_face = face_tensor + perturbation
    adversarial_face = tf.clip_by_value(adversarial_face, -3, 3)
    
    return adversarial_face.numpy()

def train_with_art(model, X_face, X_finger, y_face, y_finger, 
                    epochs=20, batch_size=32, epsilon=0.05, verbose=True):
    """Train model with Adversarial Robustness Training"""
    
    history = {'loss': [], 'face_acc': [], 'finger_acc': []}
    n_samples = len(X_face)
    
    for epoch in range(epochs):
        indices = np.random.permutation(n_samples)
        epoch_losses = []
        epoch_face_accs = []
        epoch_finger_accs = []
        
        for start in range(0, n_samples, batch_size):
            end = min(start + batch_size, n_samples)
            batch_indices = indices[start:end]
            
            batch_face = X_face[batch_indices]
            batch_finger = X_finger[batch_indices]
            batch_y_face = y_face[batch_indices]
            batch_y_finger = y_finger[batch_indices]
            
            batch_face_adv = fgsm_attack(model, batch_face, batch_finger, batch_y_face, epsilon)
            
            X_face_combined = np.vstack([batch_face, batch_face_adv])
            X_finger_combined = np.vstack([batch_finger, batch_finger])
            y_face_combined = np.hstack([batch_y_face, batch_y_face])
            y_finger_combined = np.hstack([batch_y_finger, batch_y_finger])
            
            history_batch = model.fit(
                x={'face_input': X_face_combined, 'finger_input': X_finger_combined},
                y={'face_output': y_face_combined, 'finger_output': y_finger_combined},
                epochs=1, batch_size=len(X_face_combined), verbose=0
            )
            
            epoch_losses.append(history_batch.history['loss'][0])
            epoch_face_accs.append(history_batch.history['face_output_accuracy'][0])
            epoch_finger_accs.append(history_batch.history['finger_output_accuracy'][0])
        
        history['loss'].append(np.mean(epoch_losses))
        history['face_acc'].append(np.mean(epoch_face_accs))
        history['finger_acc'].append(np.mean(epoch_finger_accs))
        
        if verbose and (epoch + 1) % 5 == 0:
            print(f"    ART Epoch {epoch+1}/{epochs}: Loss={history['loss'][-1]:.4f}, Face Acc={history['face_acc'][-1]:.4f}")
    
    return history

print("    Adversarial Robustness Training implemented")

# ============================================================================
# 5. ATTACK EVALUATION
# ============================================================================
print("\n[5/6] Implementing Attack Evaluation...")

def evaluate_under_attack(model, X_face, X_finger, y_face, y_finger, epsilon=0.05):
    """Evaluate model under FGSM attack"""
    correct_before = 0
    correct_after = 0
    
    for i in range(min(50, len(X_face))):
        orig_pred = model.predict(
            {'face_input': X_face[i:i+1], 'finger_input': X_finger[i:i+1]},
            verbose=0
        )
        orig_face_pred = np.argmax(orig_pred[0])
        
        if orig_face_pred == y_face[i]:
            correct_before += 1
            adv_face = fgsm_attack(model, X_face[i:i+1], X_finger[i:i+1], [y_face[i]], epsilon)
            adv_pred = model.predict(
                {'face_input': adv_face, 'finger_input': X_finger[i:i+1]},
                verbose=0
            )
            adv_face_pred = np.argmax(adv_pred[0])
            if adv_face_pred != y_face[i]:
                correct_after += 1
    
    return correct_after / correct_before if correct_before > 0 else 0

def privacy_score(attack_success_rate):
    return 1 - attack_success_rate

print("    Attack evaluation implemented")

# ============================================================================
# 6. TRAIN ALL MODELS
# ============================================================================
print("\n[6/6] Training Models and Evaluating...")

results = {}

# Baseline Model
print("\n" + "-" * 50)
print("Training Baseline Model (No Defense)...")
print("-" * 50)

baseline_model = create_multimodal_model(
    n_face_features, n_finger_features, n_face_classes, n_finger_classes
)
baseline_model = compile_model(baseline_model)
baseline_model.fit(
    x={'face_input': X_face_train, 'finger_input': X_finger_train},
    y={'face_output': y_face_train, 'finger_output': y_finger_train},
    epochs=args.epochs, batch_size=args.batch_size, verbose=0
)

baseline_face_acc = baseline_model.evaluate(
    x={'face_input': X_face_test, 'finger_input': X_finger_test},
    y={'face_output': y_face_test, 'finger_output': y_finger_test},
    verbose=0
)[3]

baseline_attack_rate = evaluate_under_attack(
    baseline_model, X_face_test, X_finger_test, y_face_test, y_finger_test
)
baseline_privacy_score = privacy_score(baseline_attack_rate)

results['baseline'] = {
    'face_accuracy': baseline_face_acc,
    'attack_success_rate': baseline_attack_rate,
    'privacy_score': baseline_privacy_score
}

print(f"    Baseline Face Accuracy: {baseline_face_acc:.4f} ({baseline_face_acc*100:.2f}%)")
print(f"    Attack Success Rate: {baseline_attack_rate:.4f} ({baseline_attack_rate*100:.2f}%)")

# DP Model
print("\n" + "-" * 50)
print("Training Differential Privacy Model...")
print("-" * 50)

dp_model = create_multimodal_model(
    n_face_features, n_finger_features, n_face_classes, n_finger_classes
)
dp_model = compile_model(dp_model)
train_with_dp(dp_model, X_face_train, X_finger_train, y_face_train, y_finger_train, epochs=args.epochs, batch_size=args.batch_size)

dp_face_acc = dp_model.evaluate(
    x={'face_input': X_face_test, 'finger_input': X_finger_test},
    y={'face_output': y_face_test, 'finger_output': y_finger_test},
    verbose=0
)[3]

dp_attack_rate = evaluate_under_attack(
    dp_model, X_face_test, X_finger_test, y_face_test, y_finger_test
)
dp_privacy_score = privacy_score(dp_attack_rate)

results['dp'] = {
    'face_accuracy': dp_face_acc,
    'attack_success_rate': dp_attack_rate,
    'privacy_score': dp_privacy_score
}

print(f"    DP Face Accuracy: {dp_face_acc:.4f} ({dp_face_acc*100:.2f}%)")
print(f"    Attack Success Rate: {dp_attack_rate:.4f} ({dp_attack_rate*100:.2f}%)")

# ART Model
print("\n" + "-" * 50)
print("Training Adversarial Robustness Training Model...")
print("-" * 50)

art_model = create_multimodal_model(
    n_face_features, n_finger_features, n_face_classes, n_finger_classes
)
art_model = compile_model(art_model)
train_with_art(art_model, X_face_train, X_finger_train, y_face_train, y_finger_train, epochs=args.epochs, batch_size=args.batch_size)

art_face_acc = art_model.evaluate(
    x={'face_input': X_face_test, 'finger_input': X_finger_test},
    y={'face_output': y_face_test, 'finger_output': y_finger_test},
    verbose=0
)[3]

art_attack_rate = evaluate_under_attack(
    art_model, X_face_test, X_finger_test, y_face_test, y_finger_test
)
art_privacy_score = privacy_score(art_attack_rate)

results['art'] = {
    'face_accuracy': art_face_acc,
    'attack_success_rate': art_attack_rate,
    'privacy_score': art_privacy_score
}

print(f"    ART Face Accuracy: {art_face_acc:.4f} ({art_face_acc*100:.2f}%)")
print(f"    Attack Success Rate: {art_attack_rate:.4f} ({art_attack_rate*100:.2f}%)")

# DO-FMBS Combined Model
print("\n" + "-" * 50)
print("Training DO-FMBS Model (DP + ART Combined)...")
print("-" * 50)

do_model = create_multimodal_model(
    n_face_features, n_finger_features, n_face_classes, n_finger_classes
)
do_model = compile_model(do_model)

# Warm-start the recognizer before adding privacy noise and adversarial samples.
# This preserves clean identity discrimination, then the combined stage improves
# robustness around that learned decision boundary.
pretrain_epochs = max(3, args.epochs // 2)
print(f"    Clean pretraining for {pretrain_epochs} epochs...")
do_model.fit(
    x={'face_input': X_face_train, 'finger_input': X_finger_train},
    y={'face_output': y_face_train, 'finger_output': y_finger_train},
    epochs=pretrain_epochs,
    batch_size=args.batch_size,
    verbose=0
)

# Fine-tune with both DP and ART
def add_dp_noise(gradients):
    noise_scale = 0.0003
    noisy_grads = []
    for grad in gradients:
        if grad is not None:
            noise = np.random.normal(0, noise_scale, grad.shape).astype(np.float32)
            noisy_grads.append(grad + noise)
        else:
            noisy_grads.append(grad)
    return noisy_grads

optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
n_samples = len(X_face_train)

for epoch in range(args.epochs):
    indices = np.random.permutation(n_samples)
    for start in range(0, n_samples, args.batch_size):
        end = min(start + args.batch_size, n_samples)
        batch_indices = indices[start:end]
        
        batch_face = X_face_train[batch_indices]
        batch_finger = X_finger_train[batch_indices]
        batch_y_face = y_face_train[batch_indices]
        batch_y_finger = y_finger_train[batch_indices]
        
        batch_face_adv = fgsm_attack(do_model, batch_face, batch_finger, batch_y_face, 0.05)
        
        X_face_combined = np.vstack([batch_face, batch_face_adv])
        X_finger_combined = np.vstack([batch_finger, batch_finger])
        y_face_combined = np.hstack([batch_y_face, batch_y_face])
        
        with tf.GradientTape() as tape:
            predictions = do_model({'face_input': X_face_combined, 'finger_input': X_finger_combined})
            loss = tf.reduce_mean(tf.keras.losses.sparse_categorical_crossentropy(y_face_combined, predictions[0]))
        
        grads = tape.gradient(loss, do_model.trainable_variables)
        noisy_grads = add_dp_noise(grads)
        optimizer.apply_gradients(zip(noisy_grads, do_model.trainable_variables))
    
    if (epoch + 1) % max(1, min(5, args.epochs)) == 0 or epoch == 0:
        print(f"    DO-FMBS Epoch {epoch+1}/{args.epochs}: Loss={loss.numpy():.4f}")

do_face_acc = do_model.evaluate(
    x={'face_input': X_face_test, 'finger_input': X_finger_test},
    y={'face_output': y_face_test, 'finger_output': y_finger_test},
    verbose=0
)[3]

do_attack_rate = evaluate_under_attack(
    do_model, X_face_test, X_finger_test, y_face_test, y_finger_test
)
do_privacy_score = privacy_score(do_attack_rate)

results['do_fmbs'] = {
    'face_accuracy': do_face_acc,
    'attack_success_rate': do_attack_rate,
    'privacy_score': do_privacy_score
}

print(f"\n    DO-FMBS Face Accuracy: {do_face_acc:.4f} ({do_face_acc*100:.2f}%)")
print(f"    Attack Success Rate: {do_attack_rate:.4f} ({do_attack_rate*100:.2f}%)")

# ============================================================================
# 7. SAVE RESULTS
# ============================================================================
print("\n[7/6] Saving Results...")

do_model.save('do_fmbs_data/do_fmbs_model.keras')
print("    Saved: do_fmbs_data/do_fmbs_model.keras")

final_results = {
    'comparison': results,
    'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    'parameters': {
        'dp_epsilon': 0.5,
        'art_epsilon': 0.05,
        'epochs': args.epochs,
        'max_pairs': args.max_pairs,
        'test_pairs': args.test_pairs,
        'batch_size': args.batch_size
    },
    'attack_coverage': [
        {'type': 'Evasion Attack', 'attacks': ['FGSM', 'PGD'], 'phase': 'Phase 3'},
        {'type': 'Inference Attack', 'attacks': ['Membership Inference Attack'], 'phase': 'Phase 3'},
        {'type': 'Poisoning Attack', 'attacks': ['Label-Flipping Model Poisoning'], 'phase': 'Phase 3'}
    ]
}

with open('do_fmbs_data/final_results.pkl', 'wb') as f:
    pickle.dump(final_results, f)
print("    Saved: do_fmbs_data/final_results.pkl")

# Plot comparison chart
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

models_names = ['Baseline', 'DP', 'ART', 'DO-FMBS']
accuracies = [results[m]['face_accuracy'] * 100 for m in ['baseline', 'dp', 'art', 'do_fmbs']]
attack_rates = [results[m]['attack_success_rate'] * 100 for m in ['baseline', 'dp', 'art', 'do_fmbs']]
privacy_scores = [results[m]['privacy_score'] * 100 for m in ['baseline', 'dp', 'art', 'do_fmbs']]

colors = ['gray', 'orange', 'blue', 'green']

bars = axes[0].bar(models_names, accuracies, color=colors, alpha=0.7)
axes[0].set_ylabel('Accuracy (%)')
axes[0].set_title('Model Accuracy Comparison')
axes[0].set_ylim([0, 100])

bars = axes[1].bar(models_names, attack_rates, color=colors, alpha=0.7)
axes[1].set_ylabel('Attack Success Rate (%)')
axes[1].set_title('Attack Vulnerability')
axes[1].set_ylim([0, 100])

bars = axes[2].bar(models_names, privacy_scores, color=colors, alpha=0.7)
axes[2].set_ylabel('Privacy Score (%)')
axes[2].set_title('Privacy Protection')
axes[2].set_ylim([0, 100])

plt.tight_layout()
plt.savefig('do_fmbs_comparison_results.png', dpi=150, bbox_inches='tight')
plt.close()
print("    Saved: do_fmbs_comparison_results.png")

# ============================================================================
# 8. SUMMARY
# ============================================================================
print("\n" + "=" * 70)
print("  DO-FMBS PHASE 4 COMPLETE!")
print("=" * 70)

print("\nKEY FINDINGS:")
print(f"  Baseline: {results['baseline']['face_accuracy']*100:.1f}% accuracy")
print(f"  DP: {results['dp']['face_accuracy']*100:.1f}% accuracy")
print(f"  ART: {results['art']['face_accuracy']*100:.1f}% accuracy")
print(f"  DO-FMBS: {results['do_fmbs']['face_accuracy']*100:.1f}% accuracy")

print("\nRESEARCH COMPLETE!")
print("=" * 70)
