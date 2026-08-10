"""
DO-FMBS Research Project - Phase 1: Data Loading & Preprocessing
Privacy-Preservation Model for Federated Multimodal Biometric System

Researcher: Lydiah Moraa Machora | Reg No: I152/P/9515/23
University: Jaramogi Oginga Odinga University of Science and Technology

This script:
1. Loads LFW face dataset
2. Generates synthetic fingerprint dataset (FVC-style)
3. Creates non-IID federated clients
4. Saves all data for Phase 2
"""

import numpy as np
import matplotlib.pyplot as plt
import os
from scipy import ndimage
from sklearn.datasets import fetch_lfw_people
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

# ============================================================================
# 1. LOAD FACE DATA (LFW)
# ============================================================================
print("=" * 60)
print("DO-FMBS Phase 1: Data Preparation")
print("=" * 60)

print("\n[1/4] Loading LFW Face Dataset...")
lfw = fetch_lfw_people(min_faces_per_person=20, resize=0.4, color=False)

X_face = lfw.images
y_face = lfw.target
n_face_classes = len(lfw.target_names)
n_face_features = X_face.shape[1] * X_face.shape[2]
image_shape_face = X_face.shape[1:]

print(f"    Loaded {X_face.shape[0]} images of {n_face_classes} people")
print(f"    Image size: {image_shape_face[0]} x {image_shape_face[1]} pixels")

# Flatten and normalize
X_face_flat = X_face.reshape(X_face.shape[0], n_face_features)
scaler_face = StandardScaler()
X_face_norm = scaler_face.fit_transform(X_face_flat)

# Train/test split (80/20)
X_train_face, X_test_face, y_train_face, y_test_face = train_test_split(
    X_face_norm, y_face, test_size=0.2, random_state=42, stratify=y_face
)

print(f"    Split: {X_train_face.shape[0]} train, {X_test_face.shape[0]} test")

# Save face data
os.makedirs('do_fmbs_data', exist_ok=True)
np.savez_compressed('do_fmbs_data/face_data.npz',
    X_train=X_train_face, X_test=X_test_face,
    y_train=y_train_face, y_test=y_test_face,
    n_classes=np.array([n_face_classes]),
    n_features=np.array([n_face_features]),
    image_shape=image_shape_face
)
print("    Saved: do_fmbs_data/face_data.npz")

# ============================================================================
# 2. GENERATE SYNTHETIC FINGERPRINT DATA
# ============================================================================
print("\n[2/4] Generating Synthetic Fingerprint Dataset...")

def generate_one_fingerprint(height=96, width=96, seed=None):
    """Creates one synthetic fingerprint image"""
    if seed is not None:
        np.random.seed(seed)
    
    x = np.linspace(0, 4*np.pi, width)
    y = np.linspace(0, 4*np.pi, height)
    xx, yy = np.meshgrid(x, y)
    
    angle = np.random.uniform(0, np.pi)
    freq = np.random.uniform(0.8, 1.2)
    ridges = (np.sin(freq*(xx*np.cos(angle) + yy*np.sin(angle))) +
              0.3*np.sin(2*freq*(xx*np.sin(angle) - yy*np.cos(angle))))
    
    r = np.sqrt((xx/width - 0.5)**2 + (yy/height - 0.5)**2)
    center = np.exp(-r*8) * np.sin(r*20)
    
    fp = ridges + 0.5*center
    fp = fp + np.random.normal(0, 0.15, (height, width))
    fp = ndimage.gaussian_filter(fp, sigma=0.8)
    fp = ((fp - fp.min()) / (fp.max() - fp.min()) * 255).astype(np.uint8)
    return fp

n_fingers = 100
n_impressions = 8
height, width = 96, 96

images, labels = [], []
for finger_id in range(n_fingers):
    for imp in range(n_impressions):
        seed = finger_id * 100 + imp
        fp = generate_one_fingerprint(height, width, seed)
        var = np.random.RandomState(seed).normal(0, 8, (height, width))
        fp = np.clip(fp.astype(float) + var, 0, 255).astype(np.uint8)
        images.append(fp)
        labels.append(finger_id)
    if (finger_id + 1) % 25 == 0:
        print(f"    Generated {finger_id+1}/{n_fingers} fingers...")

images = np.array(images)
labels = np.array(labels)
n_finger_classes = n_fingers
n_finger_features = height * width
image_shape_finger = (height, width)

print(f"    Generated {len(images)} fingerprint images")

# Flatten and normalize
X_finger_flat = images.reshape(images.shape[0], n_finger_features).astype(np.float64)
scaler_finger = StandardScaler()
X_finger_norm = scaler_finger.fit_transform(X_finger_flat)

# Train/test split (80/20)
X_train_finger, X_test_finger, y_train_finger, y_test_finger = train_test_split(
    X_finger_norm, labels, test_size=0.2, random_state=42, stratify=labels
)

print(f"    Split: {X_train_finger.shape[0]} train, {X_test_finger.shape[0]} test")

# Save fingerprint data
np.savez_compressed('do_fmbs_data/fingerprint_data.npz',
    X_train=X_train_finger, X_test=X_test_finger,
    y_train=y_train_finger, y_test=y_test_finger,
    n_classes=np.array([n_finger_classes]),
    n_features=np.array([n_finger_features]),
    image_shape=image_shape_finger
)
print("    Saved: do_fmbs_data/fingerprint_data.npz")

# ============================================================================
# 3. CREATE FEDERATED CLIENTS (Non-IID)
# ============================================================================
print("\n[3/4] Creating Non-IID Federated Clients...")

def non_iid_partition(X, y, n_clients=5, random_state=42):
    """Non-IID: assigns specific identity classes to each client"""
    np.random.seed(random_state)
    all_classes = np.unique(y)
    shuffled = np.random.permutation(all_classes)
    groups = np.array_split(shuffled, n_clients)
    result = []
    for g in groups:
        mask = np.isin(y, g)
        result.append({'X': X[mask], 'y': y[mask], 'n': int(mask.sum())})
    return result

face_parts = non_iid_partition(X_train_face, y_train_face, 5, 42)
finger_parts = non_iid_partition(X_train_finger, y_train_finger, 5, 42)

os.makedirs('do_fmbs_data/clients', exist_ok=True)

clients = []
for i in range(5):
    c = {
        'client_id': i,
        'face_X': face_parts[i]['X'],
        'face_y': face_parts[i]['y'],
        'face_samples': face_parts[i]['n'],
        'finger_X': finger_parts[i]['X'],
        'finger_y': finger_parts[i]['y'],
        'finger_samples': finger_parts[i]['n'],
        'total_samples': face_parts[i]['n'] + finger_parts[i]['n']
    }
    clients.append(c)
    print(f"    Client {i}: face={c['face_samples']:4d} | fingerprint={c['finger_samples']:4d} | total={c['total_samples']:4d}")

# Save each client
for c in clients:
    np.savez_compressed(
        f'do_fmbs_data/clients/client_{c["client_id"]}.npz',
        face_X=c['face_X'], face_y=c['face_y'],
        finger_X=c['finger_X'], finger_y=c['finger_y']
    )
    print(f"    Saved: do_fmbs_data/clients/client_{c['client_id']}.npz")

# ============================================================================
# 4. VISUALIZATION (Optional - saves images)
# ============================================================================
print("\n[4/4] Generating visualizations...")

# Visualize face samples
fig, axes = plt.subplots(3, 4, figsize=(14, 9))
fig.suptitle('DO-FMBS Face Modality: Sample LFW Images', fontsize=13, fontweight='bold')

for idx, ax in enumerate(axes.flat):
    if idx < len(X_train_face[:12]):
        img = X_train_face[idx].reshape(image_shape_face[0], image_shape_face[1])
        ax.imshow(img, cmap='gray')
        ax.set_title(lfw.target_names[y_train_face[idx]], fontsize=8)
        ax.axis('off')

plt.tight_layout()
plt.savefig('do_fmbs_lfw_faces.png', dpi=120, bbox_inches='tight')
plt.close()
print("    Saved: do_fmbs_lfw_faces.png")

# Visualize fingerprint samples
fig, axes = plt.subplots(3, 4, figsize=(14, 10))
fig.suptitle('DO-FMBS Fingerprint Modality: FVC-style Images\nEach row = same finger, different impressions',
             fontsize=12, fontweight='bold')

for row in range(3):
    for col in range(4):
        idx = row * 8 + col
        axes[row, col].imshow(images[idx], cmap='gray')
        axes[row, col].set_title(f'Finger {row+1}, Scan {col+1}', fontsize=9)
        axes[row, col].axis('off')

plt.tight_layout()
plt.savefig('do_fmbs_fingerprints.png', dpi=120, bbox_inches='tight')
plt.close()
print("    Saved: do_fmbs_fingerprints.png")

# Visualize client distribution
client_labels = [f'Client {c["client_id"]}' for c in clients]
face_counts = [c['face_samples'] for c in clients]
finger_counts = [c['finger_samples'] for c in clients]

x = np.arange(len(clients))
width = 0.35

fig, ax = plt.subplots(figsize=(11, 6))
b1 = ax.bar(x - width/2, face_counts, width, label='Face Samples', color='steelblue', alpha=0.85)
b2 = ax.bar(x + width/2, finger_counts, width, label='Fingerprint Samples', color='darkorange', alpha=0.85)

ax.set_xlabel('Federated Client', fontsize=12)
ax.set_ylabel('Training Samples', fontsize=12)
ax.set_title('DO-FMBS: Non-IID Federated Data Distribution\nEach client owns different identity subsets',
             fontsize=12, fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels(client_labels)
ax.legend(fontsize=11)
ax.grid(axis='y', alpha=0.3)

for bars in [b1, b2]:
    for bar in bars:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., h + 0.5, str(int(h)), ha='center', va='bottom', fontsize=9)

plt.tight_layout()
plt.savefig('do_fmbs_client_distribution.png', dpi=120, bbox_inches='tight')
plt.close()
print("    Saved: do_fmbs_client_distribution.png")

# ============================================================================
# 5. SUMMARY
# ============================================================================
print("\n" + "=" * 60)
print("  DO-FMBS PHASE 1 COMPLETE!")
print("=" * 60)

print("\n DATA SUMMARY:")
print(f"  Face Modality (LFW):")
print(f"    Identities: {n_face_classes}")
print(f"    Train/Test: {X_train_face.shape[0]} / {X_test_face.shape[0]}")
print(f"    Image size: {image_shape_face[0]} x {image_shape_face[1]} pixels")
print(f"\n  Fingerprint Modality (FVC-synthetic):")
print(f"    Identities: {n_finger_classes}")
print(f"    Train/Test: {X_train_finger.shape[0]} / {X_test_finger.shape[0]}")
print(f"    Image size: {height} x {width} pixels")
print(f"\n  Federated Clients (Non-IID):")
for c in clients:
    print(f"    Client {c['client_id']}: face={c['face_samples']} | fingerprint={c['finger_samples']} | total={c['total_samples']}")

print("\n SAVED FILES:")
print("  - do_fmbs_data/face_data.npz")
print("  - do_fmbs_data/fingerprint_data.npz")
print("  - do_fmbs_data/clients/client_*.npz")
print("  - do_fmbs_lfw_faces.png")
print("  - do_fmbs_fingerprints.png")
print("  - do_fmbs_client_distribution.png")

print("\n NEXT STEP: Run phase2_federated_learning.py")
print("=" * 60)