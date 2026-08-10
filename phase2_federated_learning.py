"""
DO-FMBS Research Project - Phase 2: Federated Learning Simulation (ENHANCED)
Privacy-Preservation Model for Federated Multimodal Biometric System

Researcher: Lydiah Moraa Machora | Reg No: I152/P/9515/23
University: Jaramogi Oginga Odinga University of Science and Technology
"""

import os
import sys
import pickle
import argparse

try:
    sys.stdout.reconfigure(line_buffering=True)
except AttributeError:
    pass

print("Starting Phase 2 backend process...", flush=True)
print("Loading TensorFlow and scientific packages...", flush=True)

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser(description="Run Phase 2 federated learning.")
parser.add_argument("--rounds", type=int, default=5)
parser.add_argument("--local-epochs", type=int, default=1)
parser.add_argument("--batch-size", type=int, default=64)
parser.add_argument("--max-pairs", type=int, default=300)
parser.add_argument("--test-pairs", type=int, default=100)
args = parser.parse_args()

# Set random seeds
tf.random.set_seed(42)
np.random.seed(42)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

print("=" * 60, flush=True)
print("DO-FMBS Phase 2: Federated Learning Simulation", flush=True)
print("=" * 60, flush=True)

# ============================================================================
# 1. LOAD DATA
# ============================================================================
print("\n[1/7] Loading Phase 1 Data...", flush=True)

face_data = np.load('do_fmbs_data/face_data.npz', allow_pickle=True)
X_train_face = face_data['X_train']
X_test_face = face_data['X_test']
y_train_face = face_data['y_train']
y_test_face = face_data['y_test']
n_face_classes = int(face_data['n_classes'])
n_face_features = int(face_data['n_features'])

finger_data = np.load('do_fmbs_data/fingerprint_data.npz', allow_pickle=True)
X_train_finger = finger_data['X_train']
X_test_finger = finger_data['X_test']
y_train_finger = finger_data['y_train']
y_test_finger = finger_data['y_test']
n_finger_classes = int(finger_data['n_classes'])
n_finger_features = int(finger_data['n_features'])

print(f"    Face: {X_train_face.shape[0]} train, {X_test_face.shape[0]} test, {n_face_classes} classes", flush=True)
print(f"    Finger: {X_train_finger.shape[0]} train, {X_test_finger.shape[0]} test, {n_finger_classes} classes", flush=True)

# ============================================================================
# 2. CREATE MATCHED PAIRS WITH DATA AUGMENTATION
# ============================================================================
print("\n[2/7] Creating Matched Face-Fingerprint Pairs...", flush=True)

# Use all fingerprint data (640 samples) and match with face data
n_pairs = min(len(X_train_finger), len(X_train_face), args.max_pairs)

# Take first n_pairs from both datasets
X_face_matched = X_train_face[:n_pairs]
y_face_matched = y_train_face[:n_pairs]
X_finger_matched = X_train_finger[:n_pairs]
y_finger_matched = y_train_finger[:n_pairs]

# Test pairs
n_test_pairs = min(len(X_test_finger), len(X_test_face), args.test_pairs)
X_face_test = X_test_face[:n_test_pairs]
y_face_test = y_test_face[:n_test_pairs]
X_finger_test = X_test_finger[:n_test_pairs]
y_finger_test = y_test_finger[:n_test_pairs]

print(f"    Training pairs: {n_pairs}", flush=True)
print(f"    Test pairs: {n_test_pairs}", flush=True)

# ============================================================================
# 3. CREATE FEDERATED CLIENTS
# ============================================================================
print("\n[3/7] Creating Federated Clients...")

def create_clients_non_iid(X_face, y_face, X_finger, y_finger, n_clients=5):
    """Create non-IID clients by distributing samples"""
    
    n_samples = len(X_face)
    indices = np.random.permutation(n_samples)
    
    # Split indices
    splits = np.array_split(indices, n_clients)
    
    clients = []
    for client_id, split in enumerate(splits):
        client = {
            'client_id': client_id,
            'face_X': X_face[split],
            'face_y': y_face[split],
            'finger_X': X_finger[split],
            'finger_y': y_finger[split],
            'n_samples': len(split)
        }
        clients.append(client)
        print(f"    Client {client_id}: {client['n_samples']} samples")
    
    return clients

clients = create_clients_non_iid(
    X_face_matched, y_face_matched,
    X_finger_matched, y_finger_matched,
    n_clients=5
)

# ============================================================================
# 4. BUILD IMPROVED MULTIMODAL MODEL
# ============================================================================
print("\n[4/7] Building Improved Multimodal Neural Network...")

def create_improved_model(face_input_dim, finger_input_dim, 
                           n_face_classes, n_finger_classes):
    """Improved model with better architecture for biometric recognition"""
    
    # Face branch
    face_input = tf.keras.Input(shape=(face_input_dim,), name='face_input')
    face_x = tf.keras.layers.Dense(256, activation='relu')(face_input)
    face_x = tf.keras.layers.BatchNormalization()(face_x)
    face_x = tf.keras.layers.Dropout(0.2)(face_x)
    face_x = tf.keras.layers.Dense(128, activation='relu')(face_x)
    face_x = tf.keras.layers.BatchNormalization()(face_x)
    face_x = tf.keras.layers.Dropout(0.2)(face_x)
    face_features = tf.keras.layers.Dense(64, activation='relu')(face_x)
    
    # Fingerprint branch
    finger_input = tf.keras.Input(shape=(finger_input_dim,), name='finger_input')
    finger_x = tf.keras.layers.Dense(256, activation='relu')(finger_input)
    finger_x = tf.keras.layers.BatchNormalization()(finger_x)
    finger_x = tf.keras.layers.Dropout(0.2)(finger_x)
    finger_x = tf.keras.layers.Dense(128, activation='relu')(finger_x)
    finger_x = tf.keras.layers.BatchNormalization()(finger_x)
    finger_x = tf.keras.layers.Dropout(0.2)(finger_x)
    finger_features = tf.keras.layers.Dense(64, activation='relu')(finger_x)
    
    # Keep the modality heads direct. Face and fingerprint labels are from
    # different identity spaces, so forcing arbitrary cross-modal pairs through
    # one fused classifier suppresses recognition accuracy.
    fused = tf.keras.layers.Concatenate()([face_features, finger_features])
    fused = tf.keras.layers.Dense(128, activation='relu')(fused)
    fused = tf.keras.layers.BatchNormalization()(fused)
    fused = tf.keras.layers.Dropout(0.3)(fused)
    fused = tf.keras.layers.Dense(64, activation='relu')(fused)
    
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

print("    Improved model defined")

# ============================================================================
# 5. FEDERATED AVERAGING WITH BETTER CONFIGURATION
# ============================================================================
print("\n[5/7] Implementing Enhanced Federated Averaging...")

class EnhancedFedAvg:
    def __init__(self, model_fn, clients_data, learning_rate=0.001):
        self.model_fn = model_fn
        self.clients_data = clients_data
        self.learning_rate = learning_rate
        self.global_model = None
        self.history = {'face_acc': [], 'finger_acc': [], 'loss': []}
        
    def initialize(self):
        self.global_model = self.model_fn()
        self.global_model = compile_model(self.global_model, self.learning_rate)
        return self.global_model
    
    def client_update(self, client, epochs=3, batch_size=32):
        """Train client for multiple epochs"""
        if client['n_samples'] == 0:
            return None
        
        local_model = self.model_fn()
        local_model = compile_model(local_model, self.learning_rate)
        local_model.set_weights(self.global_model.get_weights())
        
        face_X = client['face_X'].astype(np.float32)
        finger_X = client['finger_X'].astype(np.float32)
        
        history = local_model.fit(
            x={'face_input': face_X, 'finger_input': finger_X},
            y={'face_output': client['face_y'], 'finger_output': client['finger_y']},
            epochs=epochs,
            batch_size=batch_size,
            verbose=0
        )
        
        return {
            'weights': local_model.get_weights(),
            'n_samples': client['n_samples'],
            'face_acc': history.history['face_output_accuracy'][-1],
            'finger_acc': history.history['finger_output_accuracy'][-1],
            'loss': history.history['loss'][-1]
        }
    
    def aggregate(self, updates):
        valid_updates = [u for u in updates if u is not None]
        if not valid_updates:
            return
        
        total = sum(u['n_samples'] for u in valid_updates)
        weights = [np.zeros_like(w) for w in valid_updates[0]['weights']]
        
        for u in valid_updates:
            weight = u['n_samples'] / total
            for i, w in enumerate(u['weights']):
                weights[i] += weight * w
        
        self.global_model.set_weights(weights)
        
        # Store metrics
        self.history['face_acc'].append(np.mean([u['face_acc'] for u in valid_updates]))
        self.history['finger_acc'].append(np.mean([u['finger_acc'] for u in valid_updates]))
        self.history['loss'].append(np.mean([u['loss'] for u in valid_updates]))
    
    def train(self, rounds=20, local_epochs=3, batch_size=32):
        print(f"    Training for {rounds} rounds, {local_epochs} local epochs each...", flush=True)
        
        for r in range(rounds):
            print(f"    Round {r+1}/{rounds}: sending model to {len(self.clients_data)} clients...", flush=True)
            updates = []
            for c in self.clients_data:
                print(f"      Client {c['client_id']} training on {c['n_samples']} samples...", flush=True)
                updates.append(self.client_update(c, local_epochs, batch_size))
            self.aggregate(updates)
            
            print(f"    Round {r+1}/{rounds} complete: "
                  f"Face Acc={self.history['face_acc'][-1]:.4f}, "
                  f"Finger Acc={self.history['finger_acc'][-1]:.4f}", flush=True)
        
        return self.history

print("    Enhanced FedAvg implemented")

# ============================================================================
# 6. TRAIN THE MODEL
# ============================================================================
print("\n[6/7] Training Federated Model...")

def model_factory():
    return create_improved_model(
        n_face_features, n_finger_features,
        n_face_classes, n_finger_classes
    )

fed_avg = EnhancedFedAvg(model_factory, clients, learning_rate=0.001)
fed_avg.initialize()

history = fed_avg.train(rounds=args.rounds, local_epochs=args.local_epochs, batch_size=args.batch_size)

# ============================================================================
# 7. EVALUATE
# ============================================================================
print("\n[7/7] Evaluating on Test Data...")

final_model = fed_avg.global_model

face_test = X_face_test.astype(np.float32)
finger_test = X_finger_test.astype(np.float32)

test_results = final_model.evaluate(
    x={'face_input': face_test, 'finger_input': finger_test},
    y={'face_output': y_face_test, 'finger_output': y_finger_test},
    verbose=0
)

face_acc = test_results[3]
finger_acc = test_results[4]

print(f"\n    Face Test Accuracy: {face_acc:.4f} ({face_acc*100:.2f}%)")
print(f"    Fingerprint Test Accuracy: {finger_acc:.4f} ({finger_acc*100:.2f}%)")

# ============================================================================
# 8. CALCULATE BIOMETRIC METRICS
# ============================================================================
print("\n[8/7] Calculating Biometric Metrics...")

# Get predictions
predictions = final_model.predict({'face_input': face_test, 'finger_input': finger_test}, verbose=0)
face_preds = np.argmax(predictions[0], axis=1)
finger_preds = np.argmax(predictions[1], axis=1)

# FAR (simplified)
genuine = (face_preds == y_face_test)
FRR = 1 - np.mean(genuine)

# Simple FAR calculation
n_attempts = 0
n_false_accepts = 0
for i in range(min(100, len(face_test))):
    for j in range(min(100, len(face_test))):
        if i != j and y_face_test[i] != y_face_test[j]:
            if face_preds[i] == y_face_test[j]:
                n_false_accepts += 1
            n_attempts += 1

FAR = n_false_accepts / n_attempts if n_attempts > 0 else 0

print(f"\n    Face Recognition Accuracy: {face_acc:.2%}")
print(f"    False Acceptance Rate (FAR): {FAR:.2%}")
print(f"    False Rejection Rate (FRR): {FRR:.2%}")

# ============================================================================
# 9. SAVE RESULTS
# ============================================================================
print("\n[9/7] Saving Results...")

final_model.save('do_fmbs_data/baseline_model.keras')
print("    Saved: do_fmbs_data/baseline_model.keras")

results = {
    'face_accuracy': float(face_acc),
    'finger_accuracy': float(finger_acc),
    'FAR': float(FAR),
    'FRR': float(FRR),
    'history': history,
    'training_parameters': {
        'rounds': args.rounds,
        'local_epochs': args.local_epochs,
        'batch_size': args.batch_size
    }
}

with open('do_fmbs_data/baseline_metrics.pkl', 'wb') as f:
    pickle.dump(results, f)
print("    Saved: do_fmbs_data/baseline_metrics.pkl")

# Plot training curves
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

axes[0].plot(range(1, len(history['face_acc']) + 1), history['face_acc'], 'b-o', markersize=6)
axes[0].set_xlabel('Federated Round')
axes[0].set_ylabel('Accuracy')
axes[0].set_title('Face Recognition Accuracy')
axes[0].grid(True, alpha=0.3)
axes[0].set_ylim([0, 1])

axes[1].plot(range(1, len(history['finger_acc']) + 1), history['finger_acc'], 'r-o', markersize=6)
axes[1].set_xlabel('Federated Round')
axes[1].set_ylabel('Accuracy')
axes[1].set_title('Fingerprint Recognition Accuracy')
axes[1].grid(True, alpha=0.3)
axes[1].set_ylim([0, 1])

plt.tight_layout()
plt.savefig('do_fmbs_training_accuracy.png', dpi=120, bbox_inches='tight')
plt.close()
print("    Saved: do_fmbs_training_accuracy.png")

# ============================================================================
# 10. FINAL SUMMARY
# ============================================================================
print("\n" + "=" * 60)
print("  DO-FMBS PHASE 2 COMPLETE!")
print("=" * 60)

print("\n BASELINE PERFORMANCE SUMMARY:")
print(f"  Face Recognition Accuracy: {face_acc:.2%}")
print(f"  Fingerprint Recognition Accuracy: {finger_acc:.2%}")
print(f"  False Acceptance Rate (FAR): {FAR:.2%}")
print(f"  False Rejection Rate (FRR): {FRR:.2%}")

print("\n SAVED FILES:")
print("  - do_fmbs_data/baseline_model.keras")
print("  - do_fmbs_data/baseline_metrics.pkl")
print("  - do_fmbs_training_accuracy.png")

print("\n NEXT STEP: Phase 3 - Adversarial Attack Simulation")
print("=" * 60)
